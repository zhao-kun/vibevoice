# VibeVoice

A flexible and memory-efficient implementation of VibeVoice, a speech generation model with an AR + diffusion architecture that combines text tokenization and audio processing for high-quality text-to-speech synthesis with voice cloning capabilities.

## Overview

VibeVoice is a state-of-the-art text-to-speech (TTS) model that generates natural-sounding speech from text while preserving distinct speaker characteristics. This implementation is based on Microsoft's original VibeVoice but has been significantly enhanced for flexibility and efficiency.

### Key Features

- **Multi-speaker synthesis**: Generate speech with different voice characteristics using voice cloning
- **High-quality audio generation**: AR + diffusion architecture for natural-sounding speech
- **Flexible model loading**: Support for both directory-based and single-file model formats
- **Memory-efficient**: Float8 quantization support (FP8 E4M3FN) for reduced VRAM usage
- **Framework independence**: Decoupled from HuggingFace's PreTrainedModel for greater flexibility
- **Dynamic precision**: Automatic casting to bfloat16 during inference for optimal performance

### Why This Fork?

This fork was created to address several limitations of the original Microsoft VibeVoice implementation:

1. **Flexible Architecture**: All VibeVoice models are decoupled from HuggingFace's `PreTrainedModel` and now inherit directly from `torch.nn.Module`, providing more flexibility for customization and deployment

2. **Simplified Model Loading**: Support for loading models from single `.safetensors` files instead of requiring a complete directory structure, making model distribution and deployment easier

3. **Quantization Support**: Native support for FP8 (float8_e4m3fn) quantized models, reducing VRAM requirements by ~50% while maintaining generation quality

4. **Dynamic Precision**: Intelligent dtype casting during inference ensures optimal performance across different hardware configurations

5. **Simplified Attention**: Uses PyTorch's native SDPA (Scaled Dot Product Attention) only, removing Flash Attention dependency for better compatibility and simpler setup (with potential minor performance trade-offs)

## Installation

### Requirements

- Python >= 3.9
- PyTorch with CUDA support (recommended)
- CUDA-capable GPU (recommended for best performance)
- **Note**: This implementation uses PyTorch's native SDPA (Scaled Dot Product Attention). Flash Attention is not supported to simplify dependencies and improve compatibility.

### Install from source

```bash
git clone https://github.com/zhao-kun/vibevoice.git
cd vibevoice
pip install -e .
```

## Quick Start

### Download Pre-trained Models

Download the pre-trained models from HuggingFace:

- **BFloat16 version** (full precision): [vibevoice7b_bf16.safetensors](https://huggingface.co/zhaokun/vibevoice-large/blob/main/vibevoice7b_bf16.safetensors) (~14GB)
- **Float8 version** (quantized): [vibevoice7b_float8_e4m3fn.safetensors](https://huggingface.co/zhaokun/vibevoice-large/blob/main/vibevoice7b_float8_e4m3fn.safetensors) (~7GB)
- **Config file**: [config.json](https://huggingface.co/zhaokun/vibevoice-large/blob/main/config.json)

Place the downloaded files in a directory (e.g., `./models/converted/`).

### Basic Usage

Generate speech from a text file with a reference voice:

```bash
python demo/local_file_inference.py \
    --model_file ./models/converted/vibevoice7b_bf16.safetensors \
    --txt_path demo/text_examples/1p_pandora_box.txt \
    --speaker_names zh-007 \
    --output_dir ./outputs \
    --dtype bfloat16 \
    --cfg_scale 1.3 \
    --seed 42
```

For the float8 quantized version (lower VRAM usage):

```bash
python demo/local_file_inference.py \
    --model_file ./models/converted/vibevoice7b_float8_e4m3fn.safetensors \
    --txt_path demo/text_examples/1p_pandora_box.txt \
    --speaker_names zh-007 \
    --output_dir ./outputs \
    --dtype float8_e4m3fn \
    --cfg_scale 1.3 \
    --seed 42
```

### Using the Example Files

The repository includes example text and voice files for quick testing:

**Text Example**: `demo/text_examples/1p_pandora_box.txt`

```text
Speaker 1: 悟空你也太调皮了，我跟你说过，叫你不要乱扔东西。你看我还没说完呢，你把棍子又给扔掉了！月光宝盒是宝物，你把它扔掉会污染环境。唉，要是砸到小朋友呢，怎么办？就算没有砸到小朋友，砸到那些花花草草也是不对的呀！
```

**Voice Sample**: `demo/voices/zh-007_man.wav` (Chinese male voice)

The inference script will:

1. Parse the text file to extract speaker segments
2. Map each speaker to a voice sample from `demo/voices/`
3. Generate speech that matches the reference voice characteristics
4. Save the output to `./outputs/1p_pandora_box_generated.wav`

### Generated Audio Examples

Pre-generated audio examples are available in the `demo/outputs/` directory to demonstrate the quality of both model versions:

| Model Version | Output File | File Size | Description |
|---------------|-------------|-----------|-------------|
| **BFloat16** | [demo/outputs/pandora_box_bf16.wav](demo/outputs/pandora_box_bf16.wav) | ~1.1MB | Full precision generation |
| **Float8** | [demo/outputs/pandora_box_float8_e4m3fn.wav](demo/outputs/pandora_box_float8_e4m3fn.wav) | ~1.1MB | Quantized model generation |

Both examples demonstrate high-quality voice cloning with the reference voice (`zh-007_man.wav`) speaking the Pandora's Box text. The audio quality is comparable between BFloat16 and Float8 versions, showing that FP8 quantization maintains excellent generation quality while using significantly less VRAM.

### Command-line Arguments

- `--model_file`: Path to the model file (`.safetensors`)
- `--config`: Path to the config file, cloud be ommitted (`config.json`)
- `--txt_path`: Path to the input text file (supports `.txt` or `.json` format)
- `--speaker_names`: Speaker name(s) to map to voice files (space-separated for multiple speakers)
- `--output_dir`: Directory to save generated audio files (default: `./outputs`)
- `--device`: Device for inference: `cuda`, `mps`, or `cpu` (auto-detected by default)
- `--dtype`: Model weight dtype: `bfloat16` or `float8_e4m3fn` (default: `bfloat16`)
- `--cfg_scale`: Classifier-Free Guidance scale for generation (default: 1.3)
- `--seed`: Random seed for reproducibility (default: 42)

## Technical Details

### Float8 Quantization

This implementation uses FP8 (float8_e4m3fn) quantization to reduce memory footprint while maintaining audio quality. The quantization is handled through a custom `AutoCast` module that dynamically converts weights during inference.

#### How It Works

1. **Weight Storage**: Model weights are stored in `float8_e4m3fn` format, reducing memory by approximately 50% compared to bfloat16

2. **Dynamic Casting**: During forward passes, the `AutoCast` wrapper automatically casts float8 weights to bfloat16 precision based on input dtype:
   ```python
   class AutoCast.Linear(nn.Linear):
       def forward(self, input):
           if self.weight.dtype == torch.float8_e4m3fn:
               weight = self.weight.to(dtype=input.dtype)  # Cast to bfloat16
               return F.linear(input, weight, self.bias)
   ```

3. **Optimized Operations**: For linear layers, the implementation uses PyTorch's `torch._scaled_mm` operation when available, which provides hardware-accelerated FP8 matrix multiplication on supported GPUs (Ada/Hopper architecture)

4. **Universal Support**: The `AutoCast` module wraps all common layer types:
   - `Linear`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`
   - `LayerNorm`, `GroupNorm`, `RMSNorm`
   - `Embedding`

#### Benefits

- **50% VRAM Reduction**: FP8 models use approximately half the memory of BF16 models
- **Minimal Quality Loss**: Dynamic casting to BF16 during computation maintains generation quality
- **Hardware Acceleration**: Leverages native FP8 support on modern GPUs when available
- **Seamless Integration**: No changes needed to model architecture or inference code

### Architecture Overview

The model consists of:

- **Language Model Backbone**: Qwen-based transformer for text understanding and autoregressive generation
- **Acoustic Tokenizer**: VAE-based encoder/decoder for audio-to-latent and latent-to-audio conversion
- **Semantic Tokenizer**: Encoder for capturing semantic speech features
- **Diffusion Head**: DPM-Solver-based diffusion model for high-quality speech generation
- **Connectors**: Project acoustic and semantic features into the language model space

### Model Decoupling

Unlike the original implementation, all models inherit from `torch.nn.Module` instead of HuggingFace's `PreTrainedModel`. This provides:

- Greater flexibility for custom loading/saving logic
- Reduced dependency on HuggingFace transformers internals
- Easier integration with other frameworks and deployment pipelines
- Support for single-file model distribution

### Attention Mechanism

This implementation uses **SDPA (Scaled Dot Product Attention)** only, which is PyTorch's native attention implementation (`torch.nn.functional.scaled_dot_product_attention`). Flash Attention support has been removed to:

- Simplify installation (no need to compile Flash Attention from source)
- Improve compatibility across different CUDA versions and GPU architectures
- Reduce external dependencies
- Simplify the codebase and reduce maintenance complexity

**Performance Note**: While SDPA provides good performance, Flash Attention may offer better speed and memory efficiency on supported hardware. This trade-off prioritizes code simplicity and compatibility over maximum performance. Users requiring the absolute best performance may want to consider the original implementation with Flash Attention support.

SDPA automatically uses efficient memory-attention kernels when available and falls back to standard attention otherwise.

### Input Format

The text input should follow this format:
```
Speaker 1: [First speaker's text]
Speaker 2: [Second speaker's text]
Speaker 1: [First speaker continues...]
```

The inference script automatically parses speaker segments and maps them to voice samples.

## Model Files

This implementation supports two loading modes:

1. **Single File Mode**: Load from a single `.safetensors` file
   ```python
   model = VibeVoiceForConditionalInference.from_pretrain(
       "path/to/model.safetensors",
       config
   )
   ```

2. **Directory Mode**: Load from a directory with sharded weights
   ```python
   model = VibeVoiceForConditionalInference.from_pretrain(
       "path/to/model_directory/",
       config
   )
   ```

## Performance

Typical performance on NVIDIA RTX 4090:

| Model Version | VRAM Usage | RTF (Real-Time Factor) | Audio Quality |
|--------------|------------|------------------------|---------------|
| BFloat16     | ~14GB      | ~1.5x                  | Excellent     |
| Float8       | ~7GB       | ~1.6x                  | Excellent     |

*RTF < 1.0 means faster than real-time generation*

## Risks and Limitations

### Important Notice

**This project is intended for research and development purposes only.** Users must comply with all usage requirements and restrictions from the original Microsoft VibeVoice project.

### Risks

- **Potential for Misuse**: High-quality synthetic speech can be misused to create deepfakes, impersonation, fraud, or disinformation
- **Voice Cloning Ethics**: Voice cloning without explicit consent raises serious ethical and legal concerns
- **Inherited Biases**: The model may inherit biases, errors, or limitations from its base language model
- **Unexpected Outputs**: Generated audio may contain unexpected artifacts, inconsistencies, or inaccuracies

### Limitations

- **Language Support**: Primarily designed for English and Mandarin Chinese
- **Audio Quality**: May occasionally produce voice inconsistencies, background noise, or audio glitches
- **Multi-speaker Synthesis**: Voice characteristics may not always remain perfectly consistent
- **Not Production Ready**: This is experimental software not recommended for commercial or production use without thorough testing

### Responsible Use Guidelines

Users of this implementation must adhere to responsible AI practices:

1. **DO NOT** use this model for voice impersonation without explicit, recorded consent from the individual
2. **DO NOT** use this model to create or spread disinformation or misleading content
3. **DO NOT** use this model for fraud, scams, or malicious purposes
4. **DO NOT** violate any laws, regulations, or the original project's terms of use
5. **DO** clearly disclose when audio is AI-generated
6. **DO** respect privacy, consent, and intellectual property rights
7. **DO** use this technology ethically and responsibly

### Compliance with Original Project

This fork maintains the same ethical guidelines and usage restrictions as the original Microsoft VibeVoice project. Users are responsible for:

- Understanding and complying with Microsoft's responsible AI principles
- Following all usage restrictions and guidelines from the original project
- Ensuring their use case is appropriate for research and development
- Not using the model in ways that violate the original project's intended purpose

**By using this software, you agree to use it responsibly and in compliance with all applicable laws, regulations, and ethical guidelines.**

## Citation

If you use this implementation in your research, please cite the original VibeVoice paper:

```bibtex
@article{vibevoice2024,
  title={VibeVoice: Unified Autoregressive and Diffusion for Speech Generation},
  author={Microsoft Research},
  year={2024}
}
```

## License

This project follows the same license as the original Microsoft VibeVoice repository.

## Acknowledgments

- Original VibeVoice implementation by Microsoft Research
- Float8 casting techniques inspired by [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Model weights hosted on [HuggingFace](https://huggingface.co/zhaokun/vibevoice-large)

## Troubleshooting

### CUDA Out of Memory

Try using the float8 quantized model:
```bash
--dtype float8_e4m3fn --model_file vibevoice7b_float8_e4m3fn.safetensors
```

### Voice Mapping Issues

The inference script automatically maps speaker names to voice files in `demo/voices/`. Voice files should be named in a way that matches your speaker names (e.g., `zh-007_man.wav` for speaker name `zh-007` or `007`).

### Audio Quality Issues

Adjust the CFG scale (higher values increase adherence to conditioning):
```bash
--cfg_scale 1.5  # Try values between 1.0 and 2.0
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/zhao-kun/vibevoice/issues).
