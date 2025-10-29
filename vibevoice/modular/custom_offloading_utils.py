"""
Layer-wise CPU<->GPU offloading utilities for VRAM optimization.

This module provides automatic layer offloading for VibeVoice's Qwen2.5-7B backbone,
enabling inference on GPUs with limited VRAM by dynamically moving layers between
CPU and GPU memory during forward passes.

Key Features:
- Float8 E4M3FN aware (handles AutoCast wrapped parameters)
- Pinned memory support for faster transfers
- Prefetching for overlapped compute/transfer
- Automatic hook-based layer movement
- Memory tracking and profiling
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import logging
import time
import gc
from collections import defaultdict


@dataclass
class OffloadConfig:
    """Configuration for layer offloading strategy"""

    enabled: bool = False
    """Enable layer offloading"""

    num_layers_on_gpu: int = 8
    """Number of transformer layers to keep on GPU (default: 8)"""

    offload_prediction_head: bool = False
    """Offload prediction head (diffusion head) to CPU (saves ~3-4GB VRAM)"""

    offload_kv_cache: bool = False
    """Offload KV cache for CPU layers (aggressive memory saving)"""

    pin_memory: bool = False
    """Use pinned memory for faster CPU<->GPU transfers (disabled by default to save memory)"""

    prefetch_next_layer: bool = False
    """Prefetch next layer during current forward pass (disabled by default to save VRAM)"""

    async_transfer: bool = False
    """Use async transfers (experimental)"""

    verbose: bool = False
    """Print detailed offloading information"""


class LayerOffloader:
    """
    Manages layer-wise CPU<->GPU offloading for memory efficiency.

    Optimized for VibeVoice's architecture:
    - Handles 28 Qwen decoder layers (~310MB each in Float8)
    - Supports Float8 E4M3FN with AutoCast wrappers
    - Phase-aware offloading (prefill vs decode)
    - Prefetching for reduced latency

    Usage:
        config = OffloadConfig(enabled=True, num_layers_on_gpu=8)
        offloader = LayerOffloader(model, config, device='cuda')
        # Offloader automatically handles layer movements via hooks
    """

    def __init__(
        self,
        language_model: nn.Module,
        config: OffloadConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize layer offloader.

        Args:
            language_model: The QwenModel instance (model.language_model)
            config: Offloading configuration
            device: Target GPU device
            logger: Optional logger for diagnostics
        """
        self.language_model = language_model
        self.config = config
        self.device = device
        self.cpu_device = torch.device('cpu')
        self.logger = logger or logging.getLogger(__name__)

        # Track layer states
        self.offloaded_layers: Set[int] = set()
        self.gpu_resident_layers: Set[int] = set()

        # Track which layers have been pinned (to avoid redundant pinning)
        self.pinned_layers: Set[int] = set()

        # Hooks for automatic transfer
        self.pre_forward_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        self.post_forward_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}

        # Prefetching support
        self.next_layer_stream: Optional[torch.cuda.Stream] = None
        self.prefetch_buffer: Dict[int, Dict[str, torch.Tensor]] = {}

        # Memory tracking
        self.transfer_times: List[float] = []
        self.forward_times: Dict[int, List[float]] = defaultdict(list)
        self.total_transfers: int = 0

        # Setup offloading strategy
        if self.config.enabled:
            self._setup_offloading()

    def _setup_offloading(self):
        """Initialize offloading strategy"""
        try:
            total_layers = len(self.language_model.layers)
        except AttributeError:
            self.logger.error("Language model does not have 'layers' attribute")
            return

        # Determine which layers to offload
        num_gpu_layers = min(self.config.num_layers_on_gpu, total_layers)

        # Strategy: Keep last N layers on GPU (closer to output)
        # Reasoning: Later layers are more critical for final predictions
        # and typically have smaller activation sizes
        for i in range(total_layers):
            if i < total_layers - num_gpu_layers:
                self.offloaded_layers.add(i)
                self._move_layer_to_cpu(i)
            else:
                self.gpu_resident_layers.add(i)
                self._ensure_layer_on_gpu(i)

        # Register hooks for automatic transfers
        self._register_hooks()

        # Setup prefetch stream
        if self.config.prefetch_next_layer and torch.cuda.is_available():
            self.next_layer_stream = torch.cuda.Stream()

        self.logger.info(
            f"LayerOffloader initialized: {len(self.offloaded_layers)} layers on CPU, "
            f"{len(self.gpu_resident_layers)} layers on GPU"
        )

        if self.config.verbose:
            self.logger.info(f"  Offloaded layers: {sorted(self.offloaded_layers)}")
            self.logger.info(f"  GPU resident layers: {sorted(self.gpu_resident_layers)}")

    def _move_layer_to_cpu(self, layer_idx: int):
        """
        Move a layer from GPU to CPU (Float8-aware).

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # Move layer to CPU first
        layer.cpu()

        if self.config.pin_memory and layer_idx not in self.pinned_layers:
            # Pin memory for faster transfers (~30% speedup)
            # After layer.cpu(), all parameters are on CPU, so we can directly pin them
            for param in layer.parameters():
                # Pin the parameter data
                if param.device.type == 'cpu' and not param.data.is_pinned():
                    param.data = param.data.pin_memory()

                # Handle Float8 scale tensors (AutoCast wrapper)
                if hasattr(param, 'scale') and param.scale is not None:
                    if param.scale.device.type == 'cpu' and not param.scale.is_pinned():
                        param.scale = param.scale.pin_memory()

            # Mark this layer as pinned
            self.pinned_layers.add(layer_idx)

        if self.config.verbose:
            self.logger.debug(f"Moved layer {layer_idx} to CPU")

    def _ensure_layer_on_gpu(self, layer_idx: int):
        """
        Ensure a layer is on GPU.

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]
        layer.to(self.device)

        if self.config.verbose:
            self.logger.debug(f"Ensured layer {layer_idx} is on GPU")

    def _register_hooks(self):
        """Register pre/post forward hooks for automatic offloading"""
        for layer_idx in self.offloaded_layers:
            layer = self.language_model.layers[layer_idx]

            # Pre-forward hook: Move layer to GPU before forward
            # Note: with_kwargs=True means signature is (module, args, kwargs)
            def pre_hook(module, args, kwargs, layer_idx=layer_idx):
                return self._pre_forward_transfer(layer_idx, module, args, kwargs)

            # Post-forward hook: Move layer back to CPU after forward
            def post_hook(module, inputs, outputs, layer_idx=layer_idx):
                return self._post_forward_transfer(layer_idx, module, outputs)

            pre_handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            post_handle = layer.register_forward_hook(post_hook)

            self.pre_forward_hooks[layer_idx] = pre_handle
            self.post_forward_hooks[layer_idx] = post_handle

    def _pre_forward_transfer(self, layer_idx: int, module: nn.Module, args, kwargs):
        """
        Transfer layer to GPU before forward pass.

        Args:
            layer_idx: Layer index
            module: Layer module
            args: Forward positional arguments
            kwargs: Forward keyword arguments

        Returns:
            (args, kwargs) tuple (unchanged)
        """
        start = time.time()

        # Clear any stale CUDA cache before loading new layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check if we have prefetched this layer
        if layer_idx in self.prefetch_buffer:
            # Wait for async transfer to complete
            if self.next_layer_stream is not None:
                self.next_layer_stream.synchronize()

            # Weights are already on GPU from prefetch
            self.prefetch_buffer.pop(layer_idx)

            if self.config.verbose:
                self.logger.debug(f"Layer {layer_idx}: Using prefetched weights")
        else:
            # Synchronous transfer with non_blocking for pinned memory
            if self.config.pin_memory:
                module.to(self.device, non_blocking=True)
            else:
                module.to(self.device)

            if self.config.verbose:
                self.logger.debug(f"Layer {layer_idx}: Synchronous transfer to GPU")

        transfer_time = time.time() - start
        self.transfer_times.append(transfer_time)
        self.total_transfers += 1

        # Prefetch next layer if enabled
        if self.config.prefetch_next_layer and layer_idx + 1 in self.offloaded_layers:
            self._prefetch_layer(layer_idx + 1)

        return args, kwargs

    def _post_forward_transfer(self, layer_idx: int, module: nn.Module, outputs):
        """
        Transfer layer back to CPU after forward pass.

        Args:
            layer_idx: Layer index
            module: Layer module
            outputs: Forward outputs

        Returns:
            Outputs (unchanged)
        """
        # Move back to CPU to free VRAM immediately
        module.cpu()

        # Aggressively clear CUDA cache to ensure memory is freed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Note: We don't re-pin memory here to avoid memory duplication
        # Pinning is only done once during initial setup in _move_layer_to_cpu
        # Subsequent transfers will use the already-pinned tensors

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Moved back to CPU")

        return outputs

    def _prefetch_layer(self, layer_idx: int):
        """
        Prefetch next layer asynchronously in background.

        Args:
            layer_idx: Layer index to prefetch
        """
        if not self.config.prefetch_next_layer or self.next_layer_stream is None:
            return

        layer = self.language_model.layers[layer_idx]

        # Mark that we're prefetching this layer
        self.prefetch_buffer[layer_idx] = {}

        # Use background stream for async transfer
        with torch.cuda.stream(self.next_layer_stream):
            # Move layer to GPU in background
            layer.to(self.device, non_blocking=True)

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Started prefetch")

    def synchronize(self):
        """Synchronize all async transfers"""
        if self.next_layer_stream is not None:
            self.next_layer_stream.synchronize()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory and performance statistics.

        Returns:
            Dictionary with statistics:
            - avg_transfer_time_ms: Average transfer time in milliseconds
            - total_transfers: Total number of layer transfers
            - gpu_layers: Number of layers on GPU
            - cpu_layers: Number of layers on CPU
            - estimated_overhead_pct: Estimated performance overhead percentage
        """
        avg_transfer_ms = (
            sum(self.transfer_times) / len(self.transfer_times) * 1000
            if self.transfer_times else 0
        )

        # Estimate overhead: transfer time vs typical forward time (~20ms)
        estimated_overhead_pct = (avg_transfer_ms / 20.0) * 100 if avg_transfer_ms > 0 else 0

        return {
            'avg_transfer_time_ms': avg_transfer_ms,
            'total_transfers': self.total_transfers,
            'gpu_layers': len(self.gpu_resident_layers),
            'cpu_layers': len(self.offloaded_layers),
            'estimated_overhead_pct': estimated_overhead_pct,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics for generation metrics.

        Returns:
            Dictionary with detailed statistics including transfer overhead
        """
        total_transfer_time_ms = sum(self.transfer_times) * 1000 if self.transfer_times else 0
        avg_layer_transfer_ms = total_transfer_time_ms / len(self.transfer_times) if self.transfer_times else 0

        return {
            'total_transfer_time_ms': total_transfer_time_ms,
            'avg_layer_transfer_time_ms': avg_layer_transfer_ms,
            'total_transfers': self.total_transfers,
            'gpu_layers': len(self.gpu_resident_layers),
            'cpu_layers': len(self.offloaded_layers),
        }

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_memory_stats()

        self.logger.info("="*60)
        self.logger.info("Layer Offloading Statistics")
        self.logger.info("="*60)
        self.logger.info(f"GPU layers:              {stats['gpu_layers']}")
        self.logger.info(f"CPU layers:              {stats['cpu_layers']}")
        self.logger.info(f"Total transfers:         {stats['total_transfers']}")
        self.logger.info(f"Avg transfer time:       {stats['avg_transfer_time_ms']:.2f} ms")
        self.logger.info(f"Estimated overhead:      {stats['estimated_overhead_pct']:.1f}%")
        self.logger.info("="*60)

    def cleanup(self):
        """Remove all hooks and cleanup resources"""
        for handle in self.pre_forward_hooks.values():
            handle.remove()
        for handle in self.post_forward_hooks.values():
            handle.remove()

        self.pre_forward_hooks.clear()
        self.post_forward_hooks.clear()
        self.prefetch_buffer.clear()

        self.logger.info("LayerOffloader cleaned up")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'pre_forward_hooks') and self.pre_forward_hooks:
            self.cleanup()
