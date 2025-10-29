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
from concurrent.futures import ThreadPoolExecutor, Future
import threading


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

    pin_memory: bool = True
    """Use pinned memory for faster CPU<->GPU transfers (REQUIRED for async)"""

    prefetch_next_layer: bool = True
    """Prefetch next layer during current forward pass (HIGHLY RECOMMENDED for async)"""

    async_transfer: bool = True
    """Use async transfers with ThreadPoolExecutor (RECOMMENDED for speed)"""


    cache_clear_interval: int = 50
    """Clear CUDA cache every N layer transfers (0 = never, 50 = balanced, default: 50)"""

    verbose: bool = False
    """Print detailed offloading information"""

    profile: bool = False
    """Enable detailed profiling (prints timing breakdown)"""


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

        # Async transfer support
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.transfer_futures: Dict[int, Future] = {}
        self.transfer_lock = threading.Lock()

        # Prefetching support
        self.next_layer_stream: Optional[torch.cuda.Stream] = None
        self.prefetch_buffer: Dict[int, Dict[str, torch.Tensor]] = {}
        self.prefetch_events: Dict[int, torch.cuda.Event] = {}

        # Memory tracking
        self.transfer_times: List[float] = []
        self.forward_times: Dict[int, List[float]] = defaultdict(list)
        self.total_transfers: int = 0
        self.cache_clear_counter: int = 0

        # Profiling data
        self.profile_data: Dict[str, List[float]] = defaultdict(list)
        self.token_count: int = 0
        self.last_profile_print: float = time.time()

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

        # Setup async transfer thread pool
        if self.config.async_transfer:
            self.thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offload")

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

    def _async_move_to_gpu(self, layer_idx: int):
        """
        Asynchronously move layer to GPU using thread pool.

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # Move to GPU with non-blocking transfer (requires pinned memory)
        layer.to(self.device, non_blocking=True)

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Async GPU transfer started")

    def _async_move_to_cpu(self, layer_idx: int):
        """
        Asynchronously move layer to CPU using thread pool.

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # Move to CPU with non-blocking transfer
        layer.cpu()

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Async CPU transfer started")

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
        overall_start = time.time()

        # Wait for async transfer if it was submitted
        wait_start = time.time()
        async_used = False
        if layer_idx in self.transfer_futures:
            async_used = True
            future = self.transfer_futures.pop(layer_idx)
            future.result()  # Wait for completion

            # Wait for event if using prefetch stream
            if layer_idx in self.prefetch_events:
                event = self.prefetch_events.pop(layer_idx)
                event.synchronize()

            if self.config.verbose or self.config.profile:
                wait_time = (time.time() - wait_start) * 1000
                self.logger.info(f"Layer {layer_idx}: Async transfer wait: {wait_time:.2f}ms")
        else:
            # Fallback to synchronous transfer
            sync_start = time.time()
            if self.config.pin_memory:
                module.to(self.device, non_blocking=True)
                torch.cuda.current_stream().synchronize()
            else:
                module.to(self.device)

            if self.config.verbose or self.config.profile:
                sync_time = (time.time() - sync_start) * 1000
                self.logger.warning(f"Layer {layer_idx}: SYNC transfer (NOT ASYNC!): {sync_time:.2f}ms")

        transfer_time = time.time() - overall_start
        self.transfer_times.append(transfer_time)
        self.total_transfers += 1

        if self.config.profile:
            self.profile_data[f'layer_{layer_idx}_pre_transfer'].append(transfer_time * 1000)
            self.profile_data[f'layer_{layer_idx}_async_used'].append(1 if async_used else 0)

        # Start async transfer of next layer if enabled
        prefetch_start = time.time()
        if self.config.prefetch_next_layer and layer_idx + 1 in self.offloaded_layers:
            self._start_async_prefetch(layer_idx + 1)
            if self.config.profile:
                prefetch_time = (time.time() - prefetch_start) * 1000
                self.profile_data['prefetch_submit_time'].append(prefetch_time)

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
        post_start = time.time()

        # Submit async CPU transfer if using thread pool
        if self.config.async_transfer and self.thread_pool is not None:
            # Submit async move to CPU
            future = self.thread_pool.submit(self._async_move_to_cpu, layer_idx)
            # Don't wait - let it happen in background

            if self.config.verbose or self.config.profile:
                submit_time = (time.time() - post_start) * 1000
                self.logger.info(f"Layer {layer_idx}: Async CPU offload submitted: {submit_time:.2f}ms")
        else:
            # Synchronous move to CPU
            cpu_start = time.time()
            module.cpu()
            cpu_time = (time.time() - cpu_start) * 1000

            if self.config.verbose or self.config.profile:
                self.logger.warning(f"Layer {layer_idx}: SYNC CPU transfer: {cpu_time:.2f}ms")

        # Smart cache clearing: only clear periodically to avoid overhead
        # Clearing cache is expensive (100-500ms), so we batch it
        cache_cleared = False
        if self.config.cache_clear_interval > 0:
            self.cache_clear_counter += 1
            if self.cache_clear_counter >= self.config.cache_clear_interval:
                cache_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cache_time = (time.time() - cache_start) * 1000
                if self.config.profile and cache_time > 1.0:
                    self.logger.info(f"Cache clear took: {cache_time:.2f}ms")
                self.cache_clear_counter = 0
                cache_cleared = True

        post_time = time.time() - post_start
        if self.config.profile:
            self.profile_data[f'layer_{layer_idx}_post_transfer'].append(post_time * 1000)
            if cache_cleared:
                self.profile_data['cache_clear_count'].append(1)

        # Print profiling summary every token
        if self.config.profile and layer_idx == max(self.offloaded_layers):
            self._print_profile_summary()

        return outputs

    def _start_async_prefetch(self, layer_idx: int):
        """
        Start async prefetch of next layer using ThreadPoolExecutor.

        Args:
            layer_idx: Layer index to prefetch
        """
        if not self.config.prefetch_next_layer:
            return

        if layer_idx in self.transfer_futures:
            return  # Already submitted

        if self.config.async_transfer and self.thread_pool is not None:
            # Submit async GPU transfer
            future = self.thread_pool.submit(self._async_move_to_gpu, layer_idx)

            with self.transfer_lock:
                self.transfer_futures[layer_idx] = future

            # Create event for synchronization if using stream
            if self.next_layer_stream is not None:
                event = torch.cuda.Event()
                event.record(self.next_layer_stream)
                self.prefetch_events[layer_idx] = event

            if self.config.verbose:
                self.logger.debug(f"Layer {layer_idx}: Async prefetch submitted")
        else:
            # Fallback to old prefetch method
            self._prefetch_layer_sync(layer_idx)

    def _prefetch_layer_sync(self, layer_idx: int):
        """
        Synchronous prefetch (fallback).

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
            self.logger.debug(f"Layer {layer_idx}: Started sync prefetch")

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

    def _print_profile_summary(self):
        """Print profiling summary for last token"""
        self.token_count += 1
        current_time = time.time()

        # Print every token or every 5 seconds
        if current_time - self.last_profile_print < 5.0 and self.token_count % 10 != 0:
            return

        self.logger.info("\n" + "="*80)
        self.logger.info(f"PROFILING SUMMARY - Token {self.token_count}")
        self.logger.info("="*80)

        # Analyze async usage
        total_async = sum(self.profile_data.get(f'layer_{i}_async_used', [0])[-1]
                         for i in self.offloaded_layers if f'layer_{i}_async_used' in self.profile_data)
        total_offloaded = len(self.offloaded_layers)
        async_pct = (total_async / total_offloaded * 100) if total_offloaded > 0 else 0

        self.logger.info(f"Async usage: {total_async}/{total_offloaded} layers ({async_pct:.0f}%)")

        # Per-layer timing
        for layer_idx in self.offloaded_layers:
            pre_key = f'layer_{layer_idx}_pre_transfer'
            post_key = f'layer_{layer_idx}_post_transfer'

            if pre_key in self.profile_data and self.profile_data[pre_key]:
                pre_time = self.profile_data[pre_key][-1]
                post_time = self.profile_data[post_key][-1] if post_key in self.profile_data else 0
                async_used = self.profile_data.get(f'layer_{layer_idx}_async_used', [0])[-1]

                status = "ASYNC" if async_used else "SYNC"
                self.logger.info(f"  Layer {layer_idx:2d}: Pre={pre_time:6.2f}ms  Post={post_time:6.2f}ms  [{status}]")

        # Prefetch timing
        if 'prefetch_submit_time' in self.profile_data and self.profile_data['prefetch_submit_time']:
            avg_prefetch = sum(self.profile_data['prefetch_submit_time'][-10:]) / len(self.profile_data['prefetch_submit_time'][-10:])
            self.logger.info(f"\nAvg prefetch submit time: {avg_prefetch:.2f}ms")

        self.logger.info("="*80 + "\n")
        self.last_profile_print = current_time

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

        # Print profiling summary if enabled
        if self.config.profile and self.profile_data:
            self.logger.info("\nDetailed Profiling Data:")
            for key in sorted(self.profile_data.keys()):
                if self.profile_data[key]:
                    avg_val = sum(self.profile_data[key]) / len(self.profile_data[key])
                    self.logger.info(f"  {key}: {avg_val:.2f}ms avg")

        self.logger.info("="*60)

    def cleanup(self):
        """Remove all hooks and cleanup resources"""
        # Shutdown thread pool
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

        for handle in self.pre_forward_hooks.values():
            handle.remove()
        for handle in self.post_forward_hooks.values():
            handle.remove()

        self.pre_forward_hooks.clear()
        self.post_forward_hooks.clear()
        self.prefetch_buffer.clear()
        self.transfer_futures.clear()
        self.prefetch_events.clear()

        self.logger.info("LayerOffloader cleaned up")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'pre_forward_hooks') and self.pre_forward_hooks:
            self.cleanup()
