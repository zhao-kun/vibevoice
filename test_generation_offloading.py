#!/usr/bin/env python3
"""
Test script for voice generation with layer offloading support.

This script tests the complete voice generation pipeline with different
offloading configurations to find the optimal balance between VRAM usage
and generation speed.

Usage:
    # Auto-detect optimal configuration (recommended)
    python test_generation_offloading.py \\
        --txt_path demo/text_examples/2p_short.txt \\
        --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \\
        --auto

    # Use preset configuration
    python test_generation_offloading.py \\
        --txt_path demo/text_examples/2p_short.txt \\
        --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \\
        --preset consumer

    # Manual configuration
    python test_generation_offloading.py \\
        --txt_path demo/text_examples/2p_short.txt \\
        --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \\
        --num-gpu-layers 8

    # Test without offloading (baseline)
    python test_generation_offloading.py \\
        --txt_path demo/text_examples/2p_short.txt \\
        --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \\
        --no-offload

    # Benchmark multiple configurations
    python test_generation_offloading.py \\
        --txt_path demo/text_examples/2p_short.txt \\
        --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \\
        --benchmark
"""

import argparse
import os
import re
import sys
import time
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference
from vibevoice.modular.adaptive_offload import AdaptiveOffloadManager
from vibevoice.modular.custom_offloading_utils import OffloadConfig
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from config.configuration_vibevoice import VibeVoiceConfig, DEFAULT_CONFIG
from util.rand_init import get_generator

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for voice generation performance."""
    generation_time: float
    audio_duration: float
    rtf: float  # Real-time factor
    input_tokens: int
    generated_tokens: int
    total_tokens: int
    vram_used_gb: float
    vram_available_gb: float
    offload_config: Optional[OffloadConfig] = None
    transfer_overhead_ms: float = 0.0
    avg_layer_transfer_ms: float = 0.0


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse txt script content and extract speakers and their text.
    Format: Speaker 1, Speaker 2, Speaker 3, Speaker 4
    Returns: (scripts, speaker_numbers)
    """
    lines = txt_content.strip().split('\n')
    scripts = []
    speaker_numbers = []

    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)

            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        else:
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)

    return scripts, speaker_numbers


def get_vram_usage() -> Tuple[float, float]:
    """Get current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return used, total


def load_model(model_file: str, config_path: str, dtype: torch.dtype,
               attn_implementation: str, device: str,
               offload_config: Optional[OffloadConfig] = None):
    """Load model with optional offloading."""
    config_dict = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import json
            config_dict = json.load(f)
    else:
        print(f"Using default configuration: {DEFAULT_CONFIG}")
        config_dict = DEFAULT_CONFIG

    config = VibeVoiceConfig.from_dict(
        config_dict,
        torch_dtype=dtype,
        device_map="cuda" if device == "cuda" else None,
        attn_implementation=attn_implementation
    )

    # Print offload configuration
    if offload_config and offload_config.enabled:
        print("\n" + "="*70)
        print("OFFLOAD CONFIGURATION")
        print("="*70)
        print(f"Enabled: {offload_config.enabled}")
        print(f"Layers on GPU: {offload_config.num_layers_on_gpu}")
        print(f"Pin memory: {offload_config.pin_memory}")
        print(f"Prefetch next layer: {offload_config.prefetch_next_layer}")
        cache_msg = "Never" if offload_config.cache_clear_interval == 0 else f"Every {offload_config.cache_clear_interval} transfers"
        print(f"Cache clearing: {cache_msg}")
        print("="*70 + "\n")

    # Load model with device-specific logic
    model = VibeVoiceForConditionalInference.from_pretrain(
        model_file,
        config,
        device=device,
        offload_config=offload_config
    )

    return model


def run_generation(args, offload_config: Optional[OffloadConfig] = None) -> GenerationMetrics:
    """Run voice generation with given configuration."""

    # Set random seed
    get_generator(args.seed)

    # Read and parse txt file
    if not os.path.exists(args.txt_path):
        raise FileNotFoundError(f"Script file not found: {args.txt_path}")

    print(f"\nReading script from: {args.txt_path}")
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    scripts, speaker_numbers = parse_txt_script(txt_content)
    if not scripts:
        raise ValueError("No valid speaker scripts found in the txt file")

    print(f"Found {len(scripts)} speaker segments")

    # Get unique speakers
    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    # Validate voice files
    num_unique_speakers = len(unique_speaker_numbers)
    num_voice_files = len(args.voice_files)

    if num_voice_files < num_unique_speakers:
        raise ValueError(
            f"Not enough voice files provided. Script has {num_unique_speakers} speakers "
            f"but only {num_voice_files} voice files were provided."
        )

    # Map speaker numbers to voice files (in order)
    voice_samples = []
    for i, speaker_num in enumerate(unique_speaker_numbers):
        voice_path = args.voice_files[i]
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        voice_samples.append(voice_path)
        print(f"Speaker {speaker_num} -> Voice: {os.path.basename(voice_path)}")

    # Prepare inputs
    full_script = '\n'.join(scripts)
    full_script = full_script.replace("'", "'")

    print("\nLoading processor...")
    processor = VibeVoiceProcessor.from_pretrained(None)

    inputs = processor(
        text=[full_script],
        voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    # Move to device
    target_device = args.device if args.device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    # Decide dtype
    load_dtype = torch.bfloat16
    if args.dtype == "float8_e4m3fn":
        load_dtype = torch.float8_e4m3fn

    print("\nLoading model...")
    print(f"Device: {args.device}, dtype: {load_dtype}, attn: {args.attn_implementation}")

    model = load_model(
        args.model_file,
        args.config,
        load_dtype,
        args.attn_implementation,
        args.device,
        offload_config
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    # Record VRAM before generation
    vram_before, vram_total = get_vram_usage()
    print(f"\nVRAM before generation: {vram_before:.2f} GB / {vram_total:.2f} GB")

    # Generate audio
    print(f"\nStarting generation with cfg_scale: {args.cfg_scale}")
    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
    )

    generation_time = time.time() - start_time

    # Record VRAM after generation
    vram_after, _ = get_vram_usage()
    print(f"VRAM after generation: {vram_after:.2f} GB / {vram_total:.2f} GB")

    # Calculate metrics
    sample_rate = 24000
    audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
    audio_duration = audio_samples / sample_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

    input_tokens = inputs['input_ids'].shape[1]
    output_tokens = outputs.sequences.shape[1]
    generated_tokens = output_tokens - input_tokens

    # Get offloading statistics
    transfer_overhead_ms = 0.0
    avg_layer_transfer_ms = 0.0
    if model.offloader is not None:
        stats = model.offloader.get_stats()
        transfer_overhead_ms = stats['total_transfer_time_ms']
        avg_layer_transfer_ms = stats['avg_layer_transfer_time_ms']
        model.offloader.print_stats()

    # Save output
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_dir = args.output_dir
    if offload_config and offload_config.enabled:
        output_dir = os.path.join(args.output_dir, f"offload_{offload_config.num_layers_on_gpu}layers")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{txt_filename}_generated.wav")
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
    print(f"\nSaved output to {output_path}")

    return GenerationMetrics(
        generation_time=generation_time,
        audio_duration=audio_duration,
        rtf=rtf,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=output_tokens,
        vram_used_gb=vram_after,
        vram_available_gb=vram_total,
        offload_config=offload_config,
        transfer_overhead_ms=transfer_overhead_ms,
        avg_layer_transfer_ms=avg_layer_transfer_ms
    )


def print_summary(metrics: GenerationMetrics, config_name: str = "Default"):
    """Print generation summary."""
    print("\n" + "="*70)
    print(f"GENERATION SUMMARY - {config_name}")
    print("="*70)

    # Performance metrics
    print(f"Generation time: {metrics.generation_time:.2f}s")
    print(f"Audio duration: {metrics.audio_duration:.2f}s")
    print(f"RTF (Real Time Factor): {metrics.rtf:.2f}x")

    # Token metrics
    print(f"\nPrefilling tokens: {metrics.input_tokens}")
    print(f"Generated tokens: {metrics.generated_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")

    # VRAM metrics
    print(f"\nVRAM used: {metrics.vram_used_gb:.2f} GB / {metrics.vram_available_gb:.2f} GB")
    print(f"VRAM utilization: {(metrics.vram_used_gb / metrics.vram_available_gb * 100):.1f}%")

    # Offloading metrics
    if metrics.offload_config and metrics.offload_config.enabled:
        print("\nOffloading enabled:")
        print(f"  Layers on GPU: {metrics.offload_config.num_layers_on_gpu}/28")
        print(f"  Layers on CPU: {28 - metrics.offload_config.num_layers_on_gpu}/28")
        print(f"  Transfer overhead: {metrics.transfer_overhead_ms:.1f}ms ({metrics.transfer_overhead_ms/1000:.1f}s)")
        print(f"  Avg layer transfer: {metrics.avg_layer_transfer_ms:.2f}ms")
        overhead_percent = (metrics.transfer_overhead_ms / 1000 / metrics.generation_time) * 100
        print(f"  Overhead percentage: {overhead_percent:.1f}%")
    else:
        print("\nOffloading: Disabled")

    print("="*70)


def benchmark_configurations(args):
    """Run benchmark across multiple offloading configurations."""
    print("\n" + "="*70)
    print("BENCHMARKING MULTIPLE CONFIGURATIONS")
    print("="*70)

    # Test configurations
    configs_to_test = []

    # No offload (baseline)
    configs_to_test.append(("No Offload", None))

    # Preset configurations
    presets = ["light", "moderate", "balanced", "aggressive"]
    for preset in presets:
        config = AdaptiveOffloadManager.get_preset_config(preset)
        configs_to_test.append((f"Preset: {preset}", config))

    # Manual configurations
    for num_layers in [20, 16, 12, 8, 4]:
        config = OffloadConfig(
            enabled=True,
            num_layers_on_gpu=num_layers,
            pin_memory=False,  # Disabled for memory safety
            prefetch_next_layer=False,  # Disabled to save VRAM
            cache_clear_interval=50,
            verbose=False
        )
        configs_to_test.append((f"Manual: {num_layers} layers", config))

    # Run tests
    results = []
    for config_name, offload_config in configs_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {config_name}")
        print(f"{'='*70}")

        try:
            metrics = run_generation(args, offload_config)
            results.append((config_name, metrics))
            print_summary(metrics, config_name)
        except torch.cuda.OutOfMemoryError:
            print(f"❌ CUDA OOM error with {config_name}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"❌ Error with {config_name}: {e}")
            continue

        # Clear cache between tests
        torch.cuda.empty_cache()
        time.sleep(2)

    # Print comparison table
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)
    print(f"{'Configuration':<25} {'VRAM (GB)':<12} {'RTF':<8} {'Time (s)':<10} {'Overhead %':<12}")
    print("-"*70)

    for config_name, metrics in results:
        overhead_pct = 0.0
        if metrics.offload_config and metrics.offload_config.enabled:
            overhead_pct = (metrics.transfer_overhead_ms / 1000 / metrics.generation_time) * 100

        print(f"{config_name:<25} {metrics.vram_used_gb:<12.2f} {metrics.rtf:<8.2f} "
              f"{metrics.generation_time:<10.2f} {overhead_pct:<12.1f}")

    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test voice generation with layer offloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--txt_path",
        type=str,
        required=True,
        help="Path to the txt file containing the script"
    )
    parser.add_argument(
        "--voice_files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to voice .wav files (one for each speaker, in order)"
    )

    # Model arguments
    parser.add_argument(
        "--model_file",
        type=str,
        default="./models/converted/vibevoice7b_bf16.safetensors",
        help="Path to the model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./models/converted/config.json",
        help="Path to the model config"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float8_e4m3fn"],
        help="Model dtype (default: bfloat16)"
    )

    # Offloading arguments (mutually exclusive)
    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect optimal offloading configuration"
    )
    offload_group.add_argument(
        "--preset",
        type=str,
        choices=["high_end", "mid_range", "consumer", "budget", "minimal",
                 "light", "moderate", "balanced", "aggressive", "extreme"],
        help="Use preset offloading configuration"
    )
    offload_group.add_argument(
        "--num-gpu-layers",
        type=int,
        help="Number of layers to keep on GPU (manual configuration)"
    )
    offload_group.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable offloading (baseline test)"
    )
    offload_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark across multiple configurations"
    )

    # Performance tuning
    parser.add_argument(
        "--cache-clear-interval",
        type=int,
        default=50,
        help="Clear CUDA cache every N transfers (0=never, 50=balanced, 100=rare). Lower = more memory safe, higher = faster. Default: 50"
    )

    # Generation arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_offloading",
        help="Directory to save output audio files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for inference: cuda | cpu"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention implementation (default: sdpa)"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.3,
        help="CFG scale for generation (default: 1.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Handle benchmark mode
    if args.benchmark:
        benchmark_configurations(args)
        return

    # Determine offload configuration
    offload_config = None
    config_name = "No Offload"

    if args.no_offload:
        offload_config = None
        config_name = "No Offload"
    elif args.auto:
        print("\n🔍 Auto-detecting optimal offload configuration...")
        use_float8 = args.dtype == "float8_e4m3fn"
        offload_config = AdaptiveOffloadManager.auto_configure(
            total_layers=28,
            batch_size=1,
            max_seq_len=4096,
            use_float8=use_float8,
            target_utilization=0.80,
            device=args.device,
            logger=logger
        )
        config_name = "Auto-detected"
    elif args.preset:
        print(f"\n🎯 Using preset: {args.preset}")
        offload_config = AdaptiveOffloadManager.get_preset_config(args.preset)
        config_name = f"Preset: {args.preset}"
    elif args.num_gpu_layers is not None:
        print(f"\n⚙️  Manual configuration: {args.num_gpu_layers} layers on GPU")
        offload_config = OffloadConfig(
            enabled=True,
            num_layers_on_gpu=args.num_gpu_layers,
            pin_memory=False,  # Disabled for memory safety
            prefetch_next_layer=False,  # Disabled to save VRAM
            cache_clear_interval=args.cache_clear_interval,
            verbose=False
        )
        config_name = f"Manual: {args.num_gpu_layers} layers"

    # Print VRAM estimates
    if offload_config and offload_config.enabled:
        use_float8 = args.dtype == "float8_e4m3fn"
        AdaptiveOffloadManager.print_vram_table(use_float8=use_float8, logger=logger)

    # Run generation
    try:
        metrics = run_generation(args, offload_config)
        print_summary(metrics, config_name)
    except torch.cuda.OutOfMemoryError:
        print("\n❌ CUDA Out of Memory Error!")
        print("Try using more aggressive offloading:")
        print("  --preset aggressive")
        print("  --num-gpu-layers 4")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
