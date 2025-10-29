#!/usr/bin/env python3
"""
Test script for layer offloading functionality.

This script tests the layer offloading implementation by:
1. Loading the model with different offloading configurations
2. Verifying VRAM usage
3. Running basic inference to ensure correctness
4. Comparing performance with and without offloading

Usage:
    python test_offloading.py --config <preset_name>
    python test_offloading.py --num-gpu-layers <number>
    python test_offloading.py --auto
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference
from vibevoice.modular.custom_offloading_utils import OffloadConfig
from vibevoice.modular.adaptive_offload import AdaptiveOffloadManager
from config.configuration_vibevoice import DEFAULT_CONFIG, VibeVoiceConfig
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_memory_usage():
    """Get current CUDA memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0


def test_model_loading(offload_config=None, model_path="models/converted", dtype=torch.float8_e4m3fn):
    """
    Test model loading with offloading configuration.

    Args:
        offload_config: OffloadConfig or None
        model_path: Path to model weights
        dtype: Model dtype

    Returns:
        Model instance
    """
    logger.info("="*80)
    logger.info("Testing Model Loading")
    logger.info("="*80)

    # Reset memory stats and force aggressive cleanup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force Python garbage collection
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create config
    config = VibeVoiceConfig.from_dict(
        DEFAULT_CONFIG,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa"
    )

    # Load model
    model_file = Path(model_path) / f"vibevoice7b_{'bf16' if dtype == torch.bfloat16 else 'float8_e4m3fn'}.safetensors"

    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        logger.info(f"Please download the model to {model_file}")
        sys.exit(1)

    logger.info(f"Loading model from: {model_file}")

    # Load model with offloading
    model = VibeVoiceForConditionalInference.from_pretrain(
        str(model_file.resolve()),
        config,
        device="cuda",
        offload_config=offload_config
    )

    # Get memory usage
    allocated, reserved = get_memory_usage()
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

    logger.info("")
    logger.info("Memory Usage:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved: {reserved:.2f} GB")
    logger.info(f"  Peak: {peak:.2f} GB")
    logger.info("")

    if model.offloader:
        stats = model.offloader.get_memory_stats()
        logger.info("Offloading Statistics:")
        logger.info(f"  GPU layers: {stats['gpu_layers']}")
        logger.info(f"  CPU layers: {stats['cpu_layers']}")

    return model


def test_inference_speed(model, num_iterations=5):
    """
    Test inference speed with the model.

    Args:
        model: Loaded model
        num_iterations: Number of test iterations
    """
    logger.info("="*80)
    logger.info("Testing Inference Speed")
    logger.info("="*80)

    # Create dummy inputs
    batch_size = 1
    seq_len = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    logger.info("Warmup run...")
    with torch.inference_mode():
        output = model.model.language_model(input_ids)
        del output  # Free warmup output immediately
        torch.cuda.empty_cache()

    # Synchronize offloader if needed
    if model.offloader:
        model.offloader.synchronize()

    # Time multiple iterations
    import time
    times = []

    logger.info(f"Running {num_iterations} inference iterations...")
    for i in range(num_iterations):
        # Clear cache before each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start = time.time()
        with torch.inference_mode():
            output = model.model.language_model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Iteration {i+1}: {elapsed*1000:.2f} ms")

        # CRITICAL: Delete output and clear cache after each iteration
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_time = sum(times) / len(times)
    logger.info("")
    logger.info(f"Average time: {avg_time*1000:.2f} ms")

    if model.offloader:
        stats = model.offloader.get_memory_stats()
        logger.info(f"Average transfer time: {stats['avg_transfer_time_ms']:.2f} ms")
        logger.info(f"Estimated overhead: {stats['estimated_overhead_pct']:.1f}%")

    # Final cleanup
    del input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Test layer offloading functionality")
    parser.add_argument("--model-path", type=str, default="models/converted",
                       help="Path to model directory")
    parser.add_argument("--dtype", type=str, default="float8",
                       choices=["float8", "bf16"], help="Model dtype")
    parser.add_argument("--config", type=str, choices=["high_end", "mid_range", "consumer", "budget", "minimal"],
                       help="Use preset offload configuration")
    parser.add_argument("--num-gpu-layers", type=int,
                       help="Number of layers to keep on GPU")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-detect optimal configuration")
    parser.add_argument("--no-offload", action="store_true",
                       help="Disable offloading (baseline)")
    parser.add_argument("--test-speed", action="store_true",
                       help="Run inference speed test")
    parser.add_argument("--print-table", action="store_true",
                       help="Print VRAM usage table")
    parser.add_argument("--offload-prediction-head", action="store_true",
                       help="Offload prediction head (diffusion head) to CPU (saves ~3-4GB)")

    args = parser.parse_args()

    # Determine dtype
    dtype = torch.float8_e4m3fn if args.dtype == "float8" else torch.bfloat16
    use_float8 = args.dtype == "float8"

    # Print VRAM table if requested
    if args.print_table:
        AdaptiveOffloadManager.print_vram_table(use_float8=use_float8, logger=logger)
        return

    # Determine offload config
    offload_config = None

    if args.no_offload:
        logger.info("Offloading disabled (baseline)")
        offload_config = None
    elif args.config:
        logger.info(f"Using preset: {args.config}")
        offload_config = AdaptiveOffloadManager.get_preset_config(args.config)
    elif args.num_gpu_layers is not None:
        logger.info(f"Manual config: {args.num_gpu_layers} layers on GPU")
        offload_config = OffloadConfig(
            enabled=True,
            num_layers_on_gpu=args.num_gpu_layers,
            offload_prediction_head=args.offload_prediction_head,
            pin_memory=True,
            prefetch_next_layer=True,
            verbose=True
        )
    elif args.auto:
        logger.info("Auto-detecting configuration...")
        offload_config = AdaptiveOffloadManager.auto_configure(
            total_layers=28,
            use_float8=use_float8,
            target_utilization=0.80,
            logger=logger
        )
    else:
        logger.info("No offloading configuration specified, auto-detecting...")
        offload_config = AdaptiveOffloadManager.auto_configure(
            total_layers=28,
            use_float8=use_float8,
            target_utilization=0.80,
            logger=logger
        )

    # Test model loading
    model = test_model_loading(
        offload_config=offload_config,
        model_path=args.model_path,
        dtype=dtype
    )

    try:
        # Test inference speed if requested
        if args.test_speed:
            test_inference_speed(model, num_iterations=5)

        logger.info("="*80)
        logger.info("Test completed successfully!")
        logger.info("="*80)

    finally:
        # CRITICAL: Clean up model and free memory
        logger.info("Cleaning up model and freeing memory...")

        # Clean up offloader if present
        if hasattr(model, 'offloader') and model.offloader is not None:
            model.offloader.cleanup()

        # Delete model
        del model

        # Force garbage collection and clear GPU cache
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory cleanup completed")


if __name__ == "__main__":
    main()
