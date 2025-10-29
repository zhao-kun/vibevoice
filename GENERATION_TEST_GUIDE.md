# Voice Generation with Offloading - Test Guide

## Quick Start

Since your RTX 3080 doesn't support float8, use bfloat16 (bf16) mode:

### 1. Basic Test (Auto-detect optimal offloading)

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --auto
```

### 2. Use Preset Configuration

For RTX 3080 (10GB VRAM), try "consumer" or "balanced" preset:

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --preset consumer
```

### 3. Manual Configuration

Based on your successful test (4 GPU layers):

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --num-gpu-layers 4
```

### 4. Baseline (No Offloading)

Test without offloading to compare performance:

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --no-offload
```

### 5. Benchmark Mode

Test multiple configurations automatically:

```bash
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --benchmark
```

## Requirements

1. **Script file** (`--txt_path`): Text file with dialog in this format:
   ```
   Speaker 1: First line of dialog

   Speaker 2: Second line of dialog

   Speaker 1: Can repeat speakers
   ```

2. **Voice files** (`--voice_files`): One .wav file per unique speaker (in order)
   - For 2 speakers: provide 2 voice files
   - For 3 speakers: provide 3 voice files
   - Files must exist and be readable

## Output

The script will:
1. Load the model with specified offloading configuration
2. Print VRAM usage estimates
3. Run voice generation
4. Save output to `outputs_offloading/` directory
5. Print detailed metrics:
   - Generation time
   - Audio duration
   - RTF (Real Time Factor - lower is faster)
   - Token counts
   - VRAM usage
   - Offloading overhead

## Preset Configurations (BF16)

| Preset     | GPU Layers | Estimated VRAM | Recommended GPUs           |
|------------|------------|----------------|----------------------------|
| light      | 20         | 18-22 GB       | RTX 3090 24GB, A100        |
| moderate   | 16         | 14-18 GB       | RTX 4080 16GB              |
| balanced   | 12         | 12-16 GB       | RTX 3080 12GB              |
| consumer   | 12         | 12-16 GB       | RTX 3080 10GB, RTX 4070    |
| aggressive | 8          | 10-14 GB       | RTX 3060 12GB              |
| extreme    | 4          | 8-10 GB        | RTX 3060 8GB               |

**For RTX 3080 10GB**: Start with `--preset balanced` or `--num-gpu-layers 8`

## Example Workflow

```bash
# 1. Check available voice files
ls demo/voices/

# 2. Check available scripts
ls demo/text_examples/

# 3. Run quick test with 4 GPU layers (your known working config)
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --num-gpu-layers 4

# 4. If successful, try auto-detection for optimal config
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --auto

# 5. Run benchmark to find best tradeoff
python test_generation_offloading.py \
    --txt_path demo/text_examples/2p_short.txt \
    --voice_files demo/voices/speaker1.wav demo/voices/speaker2.wav \
    --dtype bfloat16 \
    --benchmark
```

## Troubleshooting

### CUDA Out of Memory
- Use more aggressive offloading: `--num-gpu-layers 4`
- Or try: `--preset extreme`

### "Not enough voice files"
- Make sure number of voice files matches number of unique speakers in script
- Check script with: `grep "^Speaker" demo/text_examples/2p_short.txt`

### "Voice file not found"
- Check file paths are correct
- Use absolute paths if needed: `/full/path/to/voice.wav`

## Understanding Metrics

- **RTF**: Real Time Factor. How many times real-time the generation takes.
  - RTF = 1.0: Real-time (10s audio takes 10s to generate)
  - RTF = 2.0: 2x slower (10s audio takes 20s to generate)
  - RTF = 0.5: 2x faster (10s audio takes 5s to generate)

- **Transfer overhead**: Time spent moving layers between CPU and GPU
  - Lower is better
  - Typically 10-30% of total time with offloading

- **VRAM utilization**: Percentage of GPU memory used
  - Target: 70-85% for optimal balance
