# CineWan - Wan Transformer GGUF

## Description

This project contains a practical and functional example of how to work with Wan models 
in GGUF format, using the base pipeline from 
[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/) for loading and inference of 
[Wan 2.1-2.2](https://github.com/Wan-Video/Wan2.1) video models.

The goal is to show how to adapt a Wan model to quantized GGUF weights and run 
efficient inference, even on GPUs with limited resources.

## Performance

With GGUF model inference I have achieved, on an RTX 3090 card, an ultra-fast loading 
in **20 seconds** of the 14B Wan 2.1 Transformer, including all components (tokenizer, 
VAE, etc.), and video inference with the following characteristics in **40 seconds**:

- **Resolution**: 512x320 pixels
- **Frames**: 81 frames
- **Steps**: 4 steps
- **Model**: LightX2V distilled in GGUF format

Model used: 
[Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF)

### GGUF Advantages

The main advantage of loading models in GGUF format is the **high loading speed**, 
although there is a slight loss in definition compared to original unquantized models.

## Project Structure

### `wan_transformer_gguf.py`

Contains the adapted Wan model architecture, specifically rebuilt to be compatible 
with the GGUF weights system.

**Main features:**

- Exactly replicates VideoXFun's `WanTransformer3DModel`
- Replaces standard heavy layers (`Linear` and `Conv3d`) with GGUF-optimized 
  equivalents through `GGMLOps`
- Allows direct loading of quantized weights efficiently
- Respects architectural particularities like `AdaLayerNorm` (adaptive normalization)

### `run_gguf_t2v.py`

Example script to run text-to-video generation using the adapted architecture and 
GGUF loader.

**Features:**

- Intelligent memory management to generate high resolution/duration videos on 
  VRAM-limited GPUs
- Complete parameter configuration (prompt, size, sampler, steps, etc.)
- Automatic transparent CPU/GPU module swapping

## System Operation

### 1. Adapted Architecture (`wan_transformer_gguf.py`)

The `WanTransformerGGUF` class reproduces the original Wan model architecture, using 
layers optimized for GGUF (quantized) weights.

**Key points:**

- All layers with important weights are created with `self.ops = GGMLOps()`
- Critical normalization layers configured as `LayerNorm` without learnable weights
- Attention layers are rebuilt to use GGUF layers
- `create_wan_transformer_gguf` assigns weights by mapping each state_dict tensor

### 2. Execution Script (`run_gguf_t2v.py`)

Demonstrates the use of the adapted model in a complete text-to-video workflow.

**Intelligent memory management:**

1. Moves only transformer and text encoder to GPU to generate latents
2. Performs memory swap: frees VRAM by moving those modules to CPU
3. Uploads VAE to GPU to decode latents into final video
4. Prevents heavy modules from coinciding in VRAM simultaneously

## Usage

### 1. Preparation

You need:
- A Wan model file in `.gguf` format
- YAML configuration
- Base components (VAE, text_encoder)

### 2. Script Execution

```bash
python run_gguf_t2v.py \
  --gguf_path path/to/model.wan.gguf \
  --base_model_path path/to/base_model/ \
  --config_path path/to/config.yaml \
  --prompt "A drone flying over a futuristic city" \
  --output_path example_video.mp4 \
  --height 320 --width 512 --video_length 17 --num_inference_steps 30
```

### 3. Integration in Custom Code (Optional)

```python
from wan_transformer_gguf import create_wan_transformer_gguf
model = create_wan_transformer_gguf(config, state_dict)
```

## Responsibility Distribution

### Script `run_gguf_t2v.py` (Project Manager)
- **Orchestration and configuration**
- Knows WHERE the files are
- Decides WHAT to generate (prompt, resolution, etc.)
- Delegates loading by calling `load_gguf_pipeline`
- Manages the pipeline once loaded (moves components between CPU/GPU)

### Module `gguf_loader` (Specialized Team)
- **Loading execution**
- Knows HOW to load each component
- Executes `.from_pretrained` calls for VAE and Tokenizer
- Executes specialized logic to load Transformer from GGUF
- Assembles all pieces and returns the finished product

## ⚠️ Important about Sampler

**ATTENTION!** It is essential to use the **UniPC (Flow_Unipc)** sampler when working 
with Wan 2.1 or 2.2 models.

Do not use other samplers except the recommended ones (UniPC or, in some cases, Flow), 
because other samplers may produce:
- Noise in the resulting video
- VAE latent interpretation problems

This warning is especially important to obtain the best quality and avoid unexpected 
results.

## Important Notes

- **Compatibility**: Designed for VideoXFun (Wan 2.1 and 2.2)
- **Differences**: Identical architecture at logical level, but with specific layers 
  for quantized weights
- **Memory**: CPU/GPU swap system enables efficient inference on low-VRAM GPUs

## References

- [Wan VideoXFun](https://github.com/aigc-apps/VideoX-Fun/)
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
- GGUF optimized architecture by [@acaimari](https://github.com/acaimari)
- Based on original [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/) code 
  created by City96

## License

This project is under **Apache 2.0** license. All components used (VideoX-Fun, 
ComfyUI-GGUF, and this project's code) are under Apache 2.0 license.
