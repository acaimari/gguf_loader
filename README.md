# GGUF Loader for Wan Models V1.0

A high-performance loader for Wan video generation models in `.gguf` format, designed 
to drastically reduce VRAM and RAM consumption during inference.

## Table of Contents

- [What is gguf_loader?](#what-is-gguf_loader)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start: Integration in 3 Steps](#quick-start-integration-in-3-steps)
- [Advanced Usage: Granular Control](#advanced-usage-granular-control)
- [Compatible GGUF Models](#compatible-gguf-models)
- [Frequently Asked Questions](#frequently-asked-questions)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## What is gguf_loader?

Video generation models like the Wan family are extremely powerful, but their large 
size (especially the Transformer component) demands a very high amount of VRAM, 
limiting their use to high-end hardware.

This module solves that problem by implementing an **on-the-fly dequantization 
strategy**. It allows loading Transformer weights from a quantized `.gguf` file 
(4/5/8 bits), keeping the model in a low-memory consumption format and decoding only 
the necessary layers in VRAM at the precise moment of execution.

The result is a **drastic reduction in required peak VRAM memory**, making it possible 
to run these models on consumer GPUs.

## Key Features

- ⚡ **Massive Memory Reduction**: Reduces Transformer VRAM consumption by 70% or more, 
  depending on quantization level
- 🎮 **Consumer GPU Inference**: Enables execution of models like Wan-14B on graphics 
  cards with limited VRAM (e.g., 12-16 GB)
- 🔌 **Simple Integration**: Designed to replace standard `.safetensors` loading with 
  a single function
- 🧠 **Intelligent Memory Management**: Includes scripts demonstrating how to load 
  components sequentially to minimize memory footprint
- 🛠️ **Flexible**: Offers both "all-in-one" loading and granular functions for 
  complete control

## Performance

With GGUF model inference I have achieved, on an **RTX 3090** card, ultra-fast loading 
in **20 seconds** of the 14B Wan 2.1 Transformer, including all components (tokenizer, 
VAE, etc.), and video inference with the following characteristics in **40 seconds**:

- **Resolution**: 512x320 pixels
- **Frames**: 81 frames
- **Steps**: 4 steps
- **Model**: LightX2V distilled in GGUF format

### GGUF Advantages

The main advantage of loading models in GGUF format is the **high loading speed**, 
although there is a slight loss in definition compared to original unquantized models.

## Installation

### Prerequisites

- Python 3.8 or higher
- A project that currently loads the Wan model using `diffusers`
- A Wan model file in `.gguf` format

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/acaimari/gguf_loader.git
   cd gguf_loader
   ```

2. **Install dependencies:**
   ```bash
   pip install torch diffusers transformers imageio omegaconf-py
   pip install gguf
   ```

## Quick Start: Integration in 3 Steps

This guide assumes you already have a working project that loads a Wan model with diffusers.

### 1. Copy the Module to Your Project

Copy the complete `gguf_loader/` directory to your project root. Make sure to maintain 
the folder structure to avoid import errors.

### 2. Locate Your Current Loading Code

Find the part of your code where you assemble the pipeline. It probably looks like this:

**PREVIOUS Code (loading with .safetensors):**
```python
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKL

# ... Loading VAE, Tokenizer, Scheduler ...

print("Loading Transformer from safetensors...")
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer"
)

pipe = WanPipeline(
    transformer=transformer,
    vae=vae,
    # ... other components
)
```

### 3. Replace Loading with `load_gguf_pipeline`

**NEW Code (loading with .gguf):**
```python
from gguf_loader import load_gguf_pipeline
from gguf_loader.scripts.run_gguf_t2v import FlowUniPCMultistepScheduler

# Configure your scheduler
scheduler = FlowUniPCMultistepScheduler(...)

print("Loading complete pipeline using GGUF Loader...")
pipe = load_gguf_pipeline(
    gguf_path="path/to/your/model.gguf",
    base_model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    config_path="path/to/your/config.yaml",
    scheduler=scheduler
)

print("Pipeline ready to use with GGUF!")
```

That's it! The pipeline now uses the GGUF-optimized Transformer.

## Advanced Usage: Granular Control

If you prefer to load the VAE and other components yourself and only use the module 
for the Transformer, use `load_gguf_transformer`:

```python
from gguf_loader import load_gguf_transformer
from diffusers import WanPipeline, AutoencoderKL

# 1. Load your components as usual
vae = AutoencoderKL.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", 
    subfolder="vae"
)
# ... Load Tokenizer, Text Encoder, Scheduler ...

# 2. Load ONLY the Transformer using the GGUF module
gguf_transformer = load_gguf_transformer(
    gguf_path="path/to/your/model.gguf"
)

# 3. Assemble the pipeline manually
pipe = WanPipeline(
    transformer=gguf_transformer,
    vae=vae,
    # ... rest of your components
)
```

For a complete example with intelligent memory management (CPU/GPU swap), check the 
`examples/run_gguf_t2v.py` script.

## Compatible GGUF Models

You can find Wan models in GGUF format in the following Hugging Face repositories:

- [city96](https://huggingface.co/city96) - Original models and conversions
- [QuantStack](https://huggingface.co/QuantStack) - Includes distilled LightX2V model
- [Kijai](https://huggingface.co/Kijai) - Additional variants

**Recommended model:** 
[Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF)

### ⚠️ Important about Sampler

**ATTENTION!** It is essential to use the **UniPC (Flow_Unipc)** sampler when working 
with Wan 2.1 or 2.2 models. Don't use other samplers as they may produce:

- Noise in the resulting video
- VAE latent interpretation problems

## Frequently Asked Questions

**What advantages does .gguf format have over .safetensors?**
- Better loading efficiency (up to 10x faster)
- Lower memory consumption (70% reduction)
- Support for advanced quantization
- Less dependency on external libraries

**Is it compatible with all Wan models?**
Only compatible with models trained or converted to `.gguf` format. Currently 
supports Wan 2.1 and 2.2.

**Where can I find integration examples?**
Check the `examples/` folder for complete integration examples with Wan 2.1.

**What minimum GPU do I need?**
With GGUF you can run Wan-14B models on GPUs with at least 12GB VRAM 
(like RTX 3060 12GB, RTX 4070, etc.).

## License

This project is distributed under the **Apache 2.0 license**. All components used 
are under the same license.

## Acknowledgments

This work is inspired and adapted from the excellent 
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/) project by 
[@city96](https://github.com/city96). We greatly appreciate their contribution to the 
open source community.

## Support

If you have any questions or need support, open an **issue** in this repository or 
contact [@acaimari](https://github.com/acaimari).
