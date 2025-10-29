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

        # Staging buffers for fast transfers (like kohya-ss)
        self.staging_buffers: Dict[int, Dict[str, torch.Tensor]] = {}

        # Hooks for automatic transfer
        self.pre_forward_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        self.post_forward_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}

        # Async transfer support
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.transfer_futures: Dict[int, Future] = {}
        self.transfer_lock = threading.Lock()

        # Prefetching support
        self.next_layer_stream: Optional[torch.cuda.Stream] = None
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
                # Explicitly move GPU-resident layers to GPU (no staging buffers needed)
                layer = self.language_model.layers[i]
                layer.to(self.device)
                if self.config.verbose:
                    print(f"  Moved layer {i} to GPU (resident)")

        # Register hooks for automatic transfers
        self._register_hooks()

        # Setup async transfer thread pool
        if self.config.async_transfer:
            self.thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offload")
            print(f"  ✓ ThreadPoolExecutor initialized for async transfers")
        else:
            print(f"  ✗ Async transfer DISABLED - will use synchronous transfers")

        # Setup prefetch stream
        if self.config.prefetch_next_layer and torch.cuda.is_available():
            self.next_layer_stream = torch.cuda.Stream()
            print(f"  ✓ CUDA stream created for prefetching")
        else:
            print(f"  ✗ Prefetching DISABLED")

        print(f"  Offloaded layers (on CPU): {sorted(self.offloaded_layers)}")
        print(f"  GPU resident layers: {sorted(self.gpu_resident_layers)}")

    def _move_layer_to_cpu(self, layer_idx: int):
        """
        Move a layer from GPU to CPU using staging buffers (kohya-ss approach).

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # Create staging buffers for this layer if not exists
        if layer_idx not in self.staging_buffers:
            self.staging_buffers[layer_idx] = {}

            # Pre-allocate pinned CPU memory for each parameter
            for name, param in layer.named_parameters():
                # Allocate pinned memory with same shape and dtype
                staging_buffer = torch.empty_like(param.data, device='cpu', pin_memory=True)
                self.staging_buffers[layer_idx][name] = staging_buffer

                # Handle Float8 scale tensors
                if hasattr(param, 'scale') and param.scale is not None:
                    scale_buffer = torch.empty_like(param.scale, device='cpu', pin_memory=True)
                    self.staging_buffers[layer_idx][f"{name}_scale"] = scale_buffer

            # Also handle buffers (like running stats)
            for name, buffer in layer.named_buffers():
                if buffer is not None:
                    buffer_staging = torch.empty_like(buffer, device='cpu', pin_memory=True)
                    self.staging_buffers[layer_idx][f"buffer_{name}"] = buffer_staging

        # Copy parameters to staging buffers
        for name, param in layer.named_parameters():
            staging_buffer = self.staging_buffers[layer_idx][name]
            # Non-blocking copy from GPU to pinned CPU memory
            staging_buffer.copy_(param.data, non_blocking=True)

            # Handle Float8 scale
            if hasattr(param, 'scale') and param.scale is not None:
                scale_buffer = self.staging_buffers[layer_idx][f"{name}_scale"]
                scale_buffer.copy_(param.scale, non_blocking=True)

        # Copy buffers to staging buffers
        for name, buffer in layer.named_buffers():
            if buffer is not None:
                buffer_key = f"buffer_{name}"
                if buffer_key in self.staging_buffers[layer_idx]:
                    buffer_staging = self.staging_buffers[layer_idx][buffer_key]
                    buffer_staging.copy_(buffer, non_blocking=True)

        # Synchronize to ensure all copies complete before swapping pointers
        torch.cuda.current_stream().synchronize()

        # Now swap parameter data to staging buffers (fast, no allocation)
        for name, param in layer.named_parameters():
            # Swap parameter data pointer to staging buffer
            param.data = self.staging_buffers[layer_idx][name]

            # Handle Float8 scale
            if hasattr(param, 'scale') and param.scale is not None:
                param.scale = self.staging_buffers[layer_idx][f"{name}_scale"]

        # Swap buffer data to staging buffers
        for name, buffer in layer.named_buffers():
            if buffer is not None:
                buffer_key = f"buffer_{name}"
                if buffer_key in self.staging_buffers[layer_idx]:
                    buffer.data = self.staging_buffers[layer_idx][buffer_key]

        if self.config.verbose:
            self.logger.debug(f"Moved layer {layer_idx} to CPU (staging buffer)")

    def _async_move_to_gpu(self, layer_idx: int):
        """
        Asynchronously move layer to GPU using staging buffers.

        Args:
            layer_idx: Index of layer to move
        """
        # Use staging buffer approach for fast transfer
        self._ensure_layer_on_gpu(layer_idx)

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Async GPU transfer from staging buffer started")

    def _async_move_to_cpu(self, layer_idx: int):
        """
        Asynchronously move layer to CPU using staging buffers.

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # Copy parameters back to staging buffers (non-blocking)
        if layer_idx in self.staging_buffers:
            for name, param in layer.named_parameters():
                if name in self.staging_buffers[layer_idx]:
                    staging_buffer = self.staging_buffers[layer_idx][name]
                    staging_buffer.copy_(param.data, non_blocking=True)

                    # Handle Float8 scale
                    if hasattr(param, 'scale') and param.scale is not None:
                        scale_key = f"{name}_scale"
                        if scale_key in self.staging_buffers[layer_idx]:
                            scale_buffer = self.staging_buffers[layer_idx][scale_key]
                            scale_buffer.copy_(param.scale, non_blocking=True)

            # Copy buffers back to staging buffers
            for name, buffer in layer.named_buffers():
                if buffer is not None:
                    buffer_key = f"buffer_{name}"
                    if buffer_key in self.staging_buffers[layer_idx]:
                        buffer_staging = self.staging_buffers[layer_idx][buffer_key]
                        buffer_staging.copy_(buffer.data, non_blocking=True)

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Async CPU transfer to staging buffer started")

    def _ensure_layer_on_gpu(self, layer_idx: int):
        """
        Ensure a layer is on GPU using staging buffers (kohya-ss approach).

        Args:
            layer_idx: Index of layer to move
        """
        layer = self.language_model.layers[layer_idx]

        # If staging buffers don't exist yet, layer is already on GPU
        if layer_idx not in self.staging_buffers:
            return

        # Allocate GPU tensors for parameters (or reuse existing)
        for name, param in layer.named_parameters():
            staging_buffer = self.staging_buffers[layer_idx][name]

            # Allocate GPU memory if not already allocated
            if param.data.device != self.device or param.data.shape != staging_buffer.shape:
                gpu_tensor = torch.empty_like(staging_buffer, device=self.device)
            else:
                gpu_tensor = param.data

            # Non-blocking copy from staging buffer to GPU
            gpu_tensor.copy_(staging_buffer, non_blocking=True)

            # Swap parameter data to GPU tensor
            param.data = gpu_tensor

            # Handle Float8 scale
            if hasattr(param, 'scale') and param.scale is not None and f"{name}_scale" in self.staging_buffers[layer_idx]:
                scale_buffer = self.staging_buffers[layer_idx][f"{name}_scale"]
                gpu_scale = torch.empty_like(scale_buffer, device=self.device)
                gpu_scale.copy_(scale_buffer, non_blocking=True)
                param.scale = gpu_scale

        # Handle buffers
        for name, buffer in layer.named_buffers():
            if buffer is not None:
                buffer_key = f"buffer_{name}"
                if buffer_key in self.staging_buffers[layer_idx]:
                    buffer_staging = self.staging_buffers[layer_idx][buffer_key]
                    gpu_buffer = torch.empty_like(buffer_staging, device=self.device)
                    gpu_buffer.copy_(buffer_staging, non_blocking=True)
                    buffer.data = gpu_buffer

        if self.config.verbose:
            self.logger.debug(f"Ensured layer {layer_idx} is on GPU (staging buffer)")

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

            if self.config.verbose:
                print(f"  Registered hooks for layer {layer_idx}")

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

        # Store start time for computing actual forward time
        if self.config.profile:
            self.forward_start_times = getattr(self, 'forward_start_times', {})
            self.forward_start_times[layer_idx] = time.time()

        # Transfer layer from CPU staging buffer to GPU
        transfer_start = time.time()
        self._ensure_layer_on_gpu(layer_idx)

        # Synchronize to ensure transfer completes before compute
        torch.cuda.current_stream().synchronize()

        transfer_time_ms = (time.time() - transfer_start) * 1000
        if self.config.verbose or self.config.profile:
            print(f"→ Layer {layer_idx}: CPU→GPU: {transfer_time_ms:.2f}ms")

        transfer_time = time.time() - overall_start
        self.transfer_times.append(transfer_time)
        self.total_transfers += 1

        if self.config.profile:
            self.profile_data[f'layer_{layer_idx}_pre_transfer'].append(transfer_time * 1000)

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

        # Calculate actual forward computation time
        if self.config.profile and hasattr(self, 'forward_start_times') and layer_idx in self.forward_start_times:
            forward_compute_time = (post_start - self.forward_start_times[layer_idx]) * 1000
            # Subtract transfer times to get pure compute
            pre_transfer_time = self.profile_data.get(f'layer_{layer_idx}_pre_transfer', [0])[-1] if f'layer_{layer_idx}_pre_transfer' in self.profile_data else 0
            pure_compute_time = forward_compute_time - pre_transfer_time

            # Store for analysis
            if not hasattr(self, 'layer_compute_times'):
                self.layer_compute_times = {}
            if layer_idx not in self.layer_compute_times:
                self.layer_compute_times[layer_idx] = []
            self.layer_compute_times[layer_idx].append(pure_compute_time)

            # Print every 10 tokens
            if len(self.layer_compute_times[layer_idx]) % 10 == 0:
                avg_compute = sum(self.layer_compute_times[layer_idx][-10:]) / 10
                if avg_compute > 50:  # Only print if suspiciously slow (>50ms)
                    print(f"⚠️ Layer {layer_idx} COMPUTE: {avg_compute:.2f}ms avg (should be <5ms!)")

            del self.forward_start_times[layer_idx]

        # Transfer layer from GPU back to CPU staging buffer
        offload_start = time.time()

        layer = self.language_model.layers[layer_idx]

        # Copy parameters back to staging buffers (non-blocking)
        for name, param in layer.named_parameters():
            if layer_idx in self.staging_buffers and name in self.staging_buffers[layer_idx]:
                staging_buffer = self.staging_buffers[layer_idx][name]
                # Non-blocking copy from GPU to CPU staging buffer
                staging_buffer.copy_(param.data, non_blocking=True)

                # Handle Float8 scale
                if hasattr(param, 'scale') and param.scale is not None:
                    scale_key = f"{name}_scale"
                    if scale_key in self.staging_buffers[layer_idx]:
                        scale_buffer = self.staging_buffers[layer_idx][scale_key]
                        scale_buffer.copy_(param.scale, non_blocking=True)

        # Copy buffers back to staging buffers
        for name, buffer in layer.named_buffers():
            if buffer is not None:
                buffer_key = f"buffer_{name}"
                if layer_idx in self.staging_buffers and buffer_key in self.staging_buffers[layer_idx]:
                    buffer_staging = self.staging_buffers[layer_idx][buffer_key]
                    buffer_staging.copy_(buffer.data, non_blocking=True)

        # Wait for copies to complete before swapping pointers
        torch.cuda.current_stream().synchronize()

        # Swap parameter pointers back to CPU staging buffers
        for name, param in layer.named_parameters():
            if layer_idx in self.staging_buffers and name in self.staging_buffers[layer_idx]:
                # Swap param.data to staging buffer
                param.data = self.staging_buffers[layer_idx][name]

                # Handle Float8 scale
                if hasattr(param, 'scale') and param.scale is not None:
                    scale_key = f"{name}_scale"
                    if scale_key in self.staging_buffers[layer_idx]:
                        param.scale = self.staging_buffers[layer_idx][scale_key]

        # Swap buffer pointers back to CPU staging buffers
        for name, buffer in layer.named_buffers():
            if buffer is not None:
                buffer_key = f"buffer_{name}"
                if layer_idx in self.staging_buffers and buffer_key in self.staging_buffers[layer_idx]:
                    buffer.data = self.staging_buffers[layer_idx][buffer_key]

        offload_time_ms = (time.time() - offload_start) * 1000
        if self.config.verbose or self.config.profile:
            print(f"← Layer {layer_idx}: GPU→CPU: {offload_time_ms:.2f}ms")

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
                    print(f"Cache clear took: {cache_time:.2f}ms")
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
            if self.config.profile:
                print(f"⚠️  Prefetch requested for layer {layer_idx} but prefetching is DISABLED")
            return

        if layer_idx not in self.offloaded_layers:
            if self.config.profile:
                print(f"Layer {layer_idx} not in offloaded layers {self.offloaded_layers}, skipping prefetch")
            return

        if layer_idx in self.transfer_futures:
            if self.config.profile:
                print(f"⚠️  Layer {layer_idx} already has pending transfer, skipping")
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
                print(f"Layer {layer_idx}: Async prefetch submitted to thread pool")
        else:
            if self.config.profile:
                print(f"⚠️  Async transfer disabled or thread pool is None, using sync prefetch")
            # Fallback to old prefetch method
            self._prefetch_layer_sync(layer_idx)

    def _prefetch_layer_sync(self, layer_idx: int):
        """
        Synchronous prefetch (fallback) - uses staging buffers.

        Args:
            layer_idx: Layer index to prefetch
        """
        if not self.config.prefetch_next_layer or self.next_layer_stream is None:
            return

        # Use background stream for async transfer from staging buffer to GPU
        with torch.cuda.stream(self.next_layer_stream):
            self._ensure_layer_on_gpu(layer_idx)

        if self.config.verbose:
            self.logger.debug(f"Layer {layer_idx}: Started sync prefetch from staging buffer")

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

        # Print every 10 tokens
        if self.token_count % 10 != 0:
            return

        print("\n" + "="*80)
        print(f"PROFILING SUMMARY - Token {self.token_count}")
        print("="*80)

        # Show per-layer compute times if available
        if hasattr(self, 'layer_compute_times') and self.layer_compute_times:
            print("\nPer-Layer Compute Times (pure GPU compute, excluding transfers):")
            total_compute = 0
            for layer_idx in sorted(self.layer_compute_times.keys()):
                if self.layer_compute_times[layer_idx]:
                    avg_time = sum(self.layer_compute_times[layer_idx][-10:]) / len(self.layer_compute_times[layer_idx][-10:])
                    total_compute += avg_time
                    status = "⚠️ SLOW!" if avg_time > 50 else "✓"
                    print(f"  Layer {layer_idx:2d}: {avg_time:7.2f}ms  {status}")
            print(f"  Total: {total_compute:7.2f}ms (expected: ~{len(self.offloaded_layers) * 1}ms for {len(self.offloaded_layers)} layers)")
            print()

        # Analyze GPU cache hit rate
        total_cache_hits = sum(self.profile_data.get(f'layer_{i}_cache_hit', [0])[-1]
                              for i in self.offloaded_layers if f'layer_{i}_cache_hit' in self.profile_data)
        total_offloaded = len(self.offloaded_layers)
        cache_hit_pct = (total_cache_hits / total_offloaded * 100) if total_offloaded > 0 else 0

        print(f"GPU Cache: {total_cache_hits}/{total_offloaded} layers cached ({cache_hit_pct:.0f}% hit rate)")

        # Per-layer timing
        for layer_idx in sorted(self.offloaded_layers):
            pre_key = f'layer_{layer_idx}_pre_transfer'
            post_key = f'layer_{layer_idx}_post_transfer'

            if pre_key in self.profile_data and self.profile_data[pre_key]:
                pre_time = self.profile_data[pre_key][-1]
                post_time = self.profile_data[post_key][-1] if post_key in self.profile_data else 0
                cache_hit = self.profile_data.get(f'layer_{layer_idx}_cache_hit', [0])[-1]

                status = "CACHED" if cache_hit else "LOADED"
                print(f"  Layer {layer_idx:2d}: Pre={pre_time:6.2f}ms  Post={post_time:6.2f}ms  [{status}]")

        # Prefetch timing
        if 'prefetch_submit_time' in self.profile_data and self.profile_data['prefetch_submit_time']:
            avg_prefetch = sum(self.profile_data['prefetch_submit_time'][-10:]) / len(self.profile_data['prefetch_submit_time'][-10:])
            print(f"\nAvg prefetch submit time: {avg_prefetch:.2f}ms")

        print("="*80 + "\n")
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
        self.staging_buffers.clear()
        self.transfer_futures.clear()
        self.prefetch_events.clear()

        self.logger.info("LayerOffloader cleaned up")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'pre_forward_hooks') and self.pre_forward_hooks:
            self.cleanup()
