"""
Adaptive offload management with VRAM estimation.

This module provides intelligent VRAM estimation and automatic configuration
for layer offloading based on available GPU memory.

Key Features:
- Accurate VRAM estimation for Float8 E4M3FN models
- Binary search for optimal layer configuration
- Real-time VRAM availability detection
- Preset configurations for common GPU tiers
"""

import torch
from typing import Tuple, Optional
from vibevoice.modular.custom_offloading_utils import OffloadConfig
import logging


class AdaptiveOffloadManager:
    """
    Automatically determine optimal offloading strategy based on:
    - Available VRAM
    - Model size and precision (Float8 vs BF16)
    - KV cache requirements
    - Batch size and sequence length
    """

    # Model architecture constants
    QWEN_LAYERS = 28
    QWEN_HIDDEN_SIZE = 3584
    QWEN_INTERMEDIATE_SIZE = 18944
    QWEN_NUM_KV_HEADS = 4
    QWEN_HEAD_DIM = 128

    # Memory constants (in GB)
    BASE_COMPONENTS_GB_FP8 = 0.8  # embed_tokens, tokenizers, connectors, heads
    BASE_COMPONENTS_GB_BF16 = 1.6
    LAYER_SIZE_GB_FP8 = 0.31  # ~310MB per layer in Float8
    LAYER_SIZE_GB_BF16 = 0.62  # ~620MB per layer in BF16

    @staticmethod
    def estimate_vram_usage(
        num_gpu_layers: int,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        use_float8: bool = True
    ) -> float:
        """
        Estimate VRAM usage in GB.

        Args:
            num_gpu_layers: Number of transformer layers on GPU
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            dtype: Data type for activations/KV cache (BF16 or FP16)
            use_float8: Whether model weights are in Float8 E4M3FN format

        Returns:
            Estimated VRAM usage in GB
        """

        # ===== MODEL WEIGHTS =====
        # Float8: 1 byte per param, BF16: 2 bytes per param
        if use_float8:
            base_gb = AdaptiveOffloadManager.BASE_COMPONENTS_GB_FP8
            layer_gb = AdaptiveOffloadManager.LAYER_SIZE_GB_FP8
        else:
            base_gb = AdaptiveOffloadManager.BASE_COMPONENTS_GB_BF16
            layer_gb = AdaptiveOffloadManager.LAYER_SIZE_GB_BF16

        weights_gb = base_gb + (num_gpu_layers * layer_gb)

        # ===== KV CACHE (always in BF16/FP16 at runtime) =====
        # Shape: [batch, num_kv_heads, seq_len, head_dim] * 2 (key + value)
        bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        kv_cache_bytes = (
            batch_size *
            AdaptiveOffloadManager.QWEN_NUM_KV_HEADS *
            max_seq_len *
            AdaptiveOffloadManager.QWEN_HEAD_DIM *
            2 *  # key + value
            bytes_per_element
        )
        kv_cache_gb = (kv_cache_bytes * num_gpu_layers) / (1024**3)

        # ===== ACTIVATION MEMORY (BF16 during forward) =====
        # Activations are always in BF16 regardless of weight format
        activation_gb = 1.0 if use_float8 else 1.5

        # ===== UPCAST BUFFERS (Float8 → BF16 conversions) =====
        # When using Float8, we need temporary buffers for upcasting
        upcast_buffer_gb = 0.5 if use_float8 else 0.0

        # ===== SAFETY BUFFER =====
        buffer_gb = 0.5

        total_gb = (
            weights_gb +
            kv_cache_gb +
            activation_gb +
            upcast_buffer_gb +
            buffer_gb
        )

        return total_gb

    @staticmethod
    def recommend_offload_config(
        total_layers: int = 28,
        target_vram_gb: float = 10.0,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        use_float8: bool = True,
        pin_memory: bool = False,
        prefetch: bool = False
    ) -> OffloadConfig:
        """
        Binary search to find optimal number of GPU layers.

        Args:
            total_layers: Total number of transformer layers (default: 28)
            target_vram_gb: Target VRAM usage in GB
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            use_float8: Whether model weights are in Float8 E4M3FN format
            pin_memory: Use pinned memory for transfers
            prefetch: Enable prefetching

        Returns:
            Optimal OffloadConfig
        """
        low, high = 0, total_layers
        best_num_layers = 0

        # Binary search for optimal configuration
        while low <= high:
            mid = (low + high) // 2
            estimated_vram = AdaptiveOffloadManager.estimate_vram_usage(
                num_gpu_layers=mid,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                use_float8=use_float8
            )

            if estimated_vram <= target_vram_gb:
                best_num_layers = mid
                low = mid + 1
            else:
                high = mid - 1

        return OffloadConfig(
            enabled=best_num_layers < total_layers,
            num_layers_on_gpu=best_num_layers,
            pin_memory=pin_memory,
            prefetch_next_layer=prefetch,
            offload_kv_cache=False,  # Start conservative
            verbose=False
        )

    @staticmethod
    def get_available_vram_gb(device: Optional[torch.device] = None) -> float:
        """
        Get currently available VRAM in GB.

        Args:
            device: CUDA device to query (default: cuda:0)

        Returns:
            Available VRAM in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        if device is None:
            device = torch.device('cuda:0')

        device_id = device.index if device.index is not None else 0

        # Get total and currently allocated memory
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)

        # Available = total - reserved (conservative estimate)
        free = total - reserved

        return free / (1024**3)

    @staticmethod
    def get_total_vram_gb(device: Optional[torch.device] = None) -> float:
        """
        Get total VRAM in GB.

        Args:
            device: CUDA device to query (default: cuda:0)

        Returns:
            Total VRAM in GB
        """
        if not torch.cuda.is_available():
            return 0.0

        if device is None:
            device = torch.device('cuda:0')

        device_id = device.index if device.index is not None else 0

        total = torch.cuda.get_device_properties(device_id).total_memory
        return total / (1024**3)

    @staticmethod
    def auto_configure(
        total_layers: int = 28,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        use_float8: bool = True,
        target_utilization: float = 0.80,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ) -> OffloadConfig:
        """
        Automatically configure offloading based on available VRAM.

        Args:
            total_layers: Total number of transformer layers
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            use_float8: Whether model weights are in Float8 E4M3FN format
            target_utilization: Target VRAM utilization (0.0-1.0)
            device: CUDA device to query
            logger: Optional logger

        Returns:
            Optimized OffloadConfig
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        # Get available VRAM
        available_vram = AdaptiveOffloadManager.get_available_vram_gb(device)
        total_vram = AdaptiveOffloadManager.get_total_vram_gb(device)

        if available_vram == 0.0:
            logger.warning("No CUDA device available, using CPU-only mode")
            return OffloadConfig(
                enabled=True,
                num_layers_on_gpu=0,
                pin_memory=False,
                prefetch_next_layer=False
            )

        # Target VRAM is available * target_utilization
        target_vram = available_vram * target_utilization

        logger.info(f"GPU: {torch.cuda.get_device_name(device or 0)}")
        logger.info(f"Total VRAM: {total_vram:.1f} GB")
        logger.info(f"Available VRAM: {available_vram:.1f} GB")
        logger.info(f"Target VRAM: {target_vram:.1f} GB ({target_utilization*100:.0f}% utilization)")

        # Get recommended configuration
        config = AdaptiveOffloadManager.recommend_offload_config(
            total_layers=total_layers,
            target_vram_gb=target_vram,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_float8=use_float8
        )

        # Estimate actual usage
        estimated_vram = AdaptiveOffloadManager.estimate_vram_usage(
            num_gpu_layers=config.num_layers_on_gpu,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_float8=use_float8
        )

        logger.info(f"Recommended config: {config.num_layers_on_gpu} layers on GPU")
        logger.info(f"Estimated VRAM usage: {estimated_vram:.1f} GB")

        if config.enabled:
            offloaded = total_layers - config.num_layers_on_gpu
            logger.info(f"Offloading {offloaded} layers to CPU")
        else:
            logger.info("No offloading needed")

        return config

    @staticmethod
    def print_vram_table(use_float8: bool = True, logger: Optional[logging.Logger] = None):
        """
        Print VRAM usage table for different configurations.

        Args:
            use_float8: Whether to calculate for Float8 model
            logger: Optional logger
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        model_type = "Float8 E4M3FN" if use_float8 else "BF16"

        logger.info("")
        logger.info("="*85)
        logger.info(f"VRAM Usage Estimates ({model_type} Model)")
        logger.info("="*85)
        logger.info(f"{'GPU Layers':<12} {'Weights':<12} {'KV Cache':<12} {'Other':<12} {'Total':<12} {'vs BF16':<12}")
        logger.info("-"*85)

        for num_layers in [28, 24, 20, 16, 12, 8, 4, 0]:
            vram = AdaptiveOffloadManager.estimate_vram_usage(
                num_gpu_layers=num_layers,
                use_float8=use_float8
            )
            vram_bf16 = AdaptiveOffloadManager.estimate_vram_usage(
                num_gpu_layers=num_layers,
                use_float8=False
            )

            if use_float8:
                weights = 0.8 + (num_layers * 0.31)
                layer_gb = 0.31
            else:
                weights = 1.6 + (num_layers * 0.62)
                layer_gb = 0.62

            kv = num_layers * 0.08  # ~80MB per layer
            other = 2.0  # activations + buffers

            logger.info(
                f"{num_layers:<12} {weights:<12.1f} {kv:<12.1f} {other:<12.1f} "
                f"{vram:<12.1f} {vram_bf16:<12.1f}"
            )

        logger.info("="*85)
        logger.info(f"Note: Estimates assume batch_size=1, max_seq_len=4096")
        logger.info("")

    @staticmethod
    def get_preset_config(preset: str) -> OffloadConfig:
        """
        Get preset configuration for common GPU tiers.

        Args:
            preset: One of ['high_end', 'mid_range', 'consumer', 'budget', 'minimal',
                           'light', 'moderate', 'balanced', 'aggressive', 'extreme']

        Returns:
            Preset OffloadConfig

        Raises:
            ValueError: If preset is not recognized
        """
        presets = {
            # RTX 4090 24GB, A100 40GB, H100 80GB
            'high_end': OffloadConfig(
                enabled=False,
                num_layers_on_gpu=28,
            ),

            # RTX 3090 24GB, RTX 4080 16GB, A6000 48GB
            'mid_range': OffloadConfig(
                enabled=True,
                num_layers_on_gpu=20,
                pin_memory=True,  # Faster transfers
                prefetch_next_layer=True,  # Use CUDA streams
                async_transfer=False,  # Disable ThreadPoolExecutor (causes slowdown!)
            ),

            # RTX 4070 12GB, RTX 3080 12GB
            'consumer': OffloadConfig(
                enabled=True,
                num_layers_on_gpu=12,
                pin_memory=True,  # Required for async
                prefetch_next_layer=True,  # Recommended for async
                async_transfer=True,
            ),

            # RTX 3060 12GB, RTX 4060 Ti 16GB
            'budget': OffloadConfig(
                enabled=True,
                num_layers_on_gpu=8,
                pin_memory=True,  # Required for async
                prefetch_next_layer=True,  # Recommended for async
                async_transfer=True,
            ),

            # RTX 3050 8GB, GTX 1080 8GB
            'minimal': OffloadConfig(
                enabled=True,
                num_layers_on_gpu=4,
                pin_memory=True,  # Required for async
                prefetch_next_layer=True,  # Recommended for async
                async_transfer=True,
                offload_kv_cache=False,
            ),

            # Alternative naming scheme (layer count based)
            'light': OffloadConfig(enabled=True, num_layers_on_gpu=20, pin_memory=True, prefetch_next_layer=True, async_transfer=True),
            'moderate': OffloadConfig(enabled=True, num_layers_on_gpu=16, pin_memory=True, prefetch_next_layer=True, async_transfer=True),
            'balanced': OffloadConfig(enabled=True, num_layers_on_gpu=12, pin_memory=True, prefetch_next_layer=True, async_transfer=True),
            'aggressive': OffloadConfig(enabled=True, num_layers_on_gpu=8, pin_memory=True, prefetch_next_layer=True, async_transfer=True),
            'extreme': OffloadConfig(enabled=True, num_layers_on_gpu=4, pin_memory=True, prefetch_next_layer=True, async_transfer=True),
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(presets.keys())}"
            )

        return presets[preset]
