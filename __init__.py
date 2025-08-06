# gguf_loader/__init__.py
# Adapted for standalone PyTorch usage by Antoni Caimari Caldes
#
# Based on code from ComfyUI-GGUF:
#   https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
# Original GGUF loading system created by City96
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Further modifications and PyTorch adaptation by caimari:
#   https://github.com/acaimari
#
# Copyright (c) 2024 City96, Caimari. All rights reserved.


import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from videox_fun.models import WanTransformer3DModel
from videox_fun.pipeline import WanPipeline

from .loader import gguf_sd_loader
from .ops import GGMLOps

logger = logging.getLogger(__name__)


def verify_model_integrity(model):
    """Verificar que todos los componentes críticos estén cargados"""
    logger.info("\n🔍 Verificando integridad del modelo...")
    
    issues = []
    
    # Verificar componentes principales
    if hasattr(model, 'patch_embedding'):
        if model.patch_embedding is None:
            issues.append("patch_embedding is None")
        elif hasattr(model.patch_embedding, 'weight'):
            if model.patch_embedding.weight is None:
                issues.append("patch_embedding.weight is None")
    
    # Verificar bloques (usando 'blocks' no 'transformer_blocks')
    blocks = getattr(model, 'blocks', getattr(model, 'transformer_blocks', []))
    for i, block in enumerate(blocks[:3]):  # Solo primeros 3 para debug
        # Verificar norm layers
        for norm_name in ['norm1', 'norm2', 'norm3']:
            if hasattr(block, norm_name):
                norm_layer = getattr(block, norm_name)
                if norm_layer is None:
                    issues.append(f"Block {i}: {norm_name} is None")
                elif hasattr(norm_layer, 'weight'):
                    if norm_layer.weight is None:
                        issues.append(f"Block {i}: {norm_name}.weight is None")
                    elif not isinstance(norm_layer.weight, (nn.Parameter, torch.Tensor)):
                        issues.append(f"Block {i}: {norm_name}.weight wrong type: {type(norm_layer.weight)}")
        
        # Verificar attention
        if hasattr(block, 'self_attn'):
            for param in ['q', 'k', 'v', 'o']:
                if hasattr(block.self_attn, param):
                    layer = getattr(block.self_attn, param)
                    if layer is None:
                        issues.append(f"Block {i}: self_attn.{param} is None")
                    elif hasattr(layer, 'weight') and layer.weight is None:
                        issues.append(f"Block {i}: self_attn.{param}.weight is None")
    
    if issues:
        logger.warning("❌ Problemas encontrados:")
        for issue in issues[:10]:  # Mostrar primeros 10
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ Todos los componentes críticos verificados")
    
    # Contar parámetros reales
    total_params = 0
    ggml_params = 0
    regular_params = 0
    none_params = 0
    
    for name, param in model.named_parameters():
        if param is not None:
            param_count = param.numel() if hasattr(param, 'numel') else 0
            total_params += param_count
            
            # Check if it's a GGML tensor
            if hasattr(param, 'ggml_tensor') or str(type(param)).find('GGML') >= 0:
                ggml_params += param_count
            else:
                regular_params += param_count
        else:
            none_params += 1
    
    logger.info(f"\n📊 Conteo de parámetros:")
    logger.info(f"  Total: {total_params/1e9:.2f}B")
    logger.info(f"  GGML: {ggml_params/1e9:.2f}B")
    logger.info(f"  Regular: {regular_params/1e9:.2f}B")
    logger.info(f"  None params: {none_params}")
    
    return len(issues) == 0


def load_gguf_pipeline(
    gguf_path: str,
    base_model_path: str,
    config_path: str,
    pipeline_class: type = WanPipeline,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    dequant_dtype: Optional[torch.dtype] = None,
    patch_dtype: Optional[torch.dtype] = None,
    transformer_config: Optional[Dict[str, Any]] = None,
    **pipeline_kwargs
) -> Any:
    """
    Load a Wan pipeline with GGUF transformer weights.
    """
    import time
    logger.info(f"🚀 Loading GGUF pipeline from {gguf_path}")

    # Initialize custom operations
    ops = GGMLOps()
    if dequant_dtype is not None:
        ops.Linear.dequant_dtype = dequant_dtype
    if patch_dtype is not None:
        ops.Linear.patch_dtype = patch_dtype

    # Load GGUF state dict
    logger.info("📖 Reading GGUF state dict...")
    start_gguf = time.time()
    gguf_state_dict = gguf_sd_loader(gguf_path)
    logger.info(f"⏱️ GGUF loaded in {time.time() - start_gguf:.2f}s, got {len(gguf_state_dict)} tensors")

    # Print some info about the GGUF file
    total_params = sum(t.numel() for t in gguf_state_dict.values())
    logger.info(f"📊 Total parameters in GGUF: {total_params/1e9:.2f}B")

    # DEBUG: Show sample of GGUF keys
    logger.info("\n🔍 DEBUG - Sample GGUF keys (first 30):")
    for i, key in enumerate(list(gguf_state_dict.keys())[:30]):
        tensor = gguf_state_dict[key]
        shape = tensor.shape if hasattr(tensor, 'shape') else 'unknown'
        dtype_info = tensor.tensor_type if hasattr(tensor, 'tensor_type') else 'unknown'
        logger.info(f"  {i:3d}. {key}: shape={shape}, type={dtype_info}")

    # Analyze GGUF key patterns
    logger.info("\n📊 DEBUG - GGUF Key Analysis:")
    key_prefixes = defaultdict(int)
    for key in gguf_state_dict.keys():
        prefix = key.split('.')[0] if '.' in key else key
        key_prefixes[prefix] += 1

    logger.info("  Key prefix distribution:")
    for prefix, count in sorted(key_prefixes.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {prefix}: {count} tensors")

    # Check for transformer blocks
    transformer_block_keys = [k for k in gguf_state_dict.keys() if 'transformer_blocks' in k or 'blocks' in k]
    logger.info(f"\n  Transformer block tensors found: {len(transformer_block_keys)}")
    if transformer_block_keys:
        # Analyze block numbers
        block_nums = set()
        for key in transformer_block_keys:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part.isdigit():
                    block_nums.add(int(part))
                    break
        logger.info(f"  Block numbers detected: {sorted(block_nums)[:20]}...")
        logger.info(f"  Total unique blocks: {len(block_nums)}")
        
        # Check for norm layers specifically
        norm_keys = [k for k in gguf_state_dict.keys() if 'norm' in k.lower()]
        logger.info(f"  Norm layer tensors found: {len(norm_keys)}")
        if norm_keys:
            logger.info("  Sample norm keys:")
            for key in norm_keys[:5]:
                logger.info(f"    - {key}")

    # Load config
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)

    # Load components manually
    logger.info(f"\n📦 Loading base components from {base_model_path}")

    # Load VAE
    start_vae = time.time()
    from videox_fun.models import AutoencoderKLWan
    vae_path = os.path.join(base_model_path, config['vae_kwargs'].get('vae_subpath', 'Wan2.1_VAE.pth'))
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(dtype)
    logger.info(f"⏱️ VAE loaded in {time.time() - start_vae:.2f}s")

    # Load Text Encoder
    start_text_enc = time.time()
    from videox_fun.models import WanT5EncoderModel, AutoTokenizer
    text_encoder_path = os.path.join(base_model_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'models_t5_umt5-xxl-enc-bf16.pth'))
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    logger.info(f"⏱️ Text encoder loaded in {time.time() - start_text_enc:.2f}s")

    # Load Tokenizer
    start_tokenizer = time.time()
    tokenizer_path = os.path.join(base_model_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'google/umt5-xxl'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"⏱️ Tokenizer loaded in {time.time() - start_tokenizer:.2f}s")

    # Create transformer with GGML operations
    logger.info("\n🔧 Creating transformer with GGML operations...")
    start_transformer = time.time()

    # Use provided config for 14B model or default config
    if transformer_config is None:
        # Default config for 14B model
        transformer_config = {
            "dim": 5120,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "in_channels": 16,
            "out_dim": 16,
            "out_channels": 16,
            "model_type": "t2v",
            "num_heads": 40,
            "num_layers": 40,
            "text_len": 512,
            "text_dim": 4096,
            "patch_size": (1, 2, 2),
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
        }
    else:
        # Asegurar que config tenga todos los campos necesarios
        if 'in_channels' not in transformer_config:
            transformer_config['in_channels'] = transformer_config.get('in_dim', 16)
        if 'out_channels' not in transformer_config:
            transformer_config['out_channels'] = transformer_config.get('out_dim', 16)

    # Import our GGUF transformer
    logger.info("📋 Creating GGUF transformer...")

    # DEBUG: Before creating transformer, analyze what we need vs what we have
    logger.info("\n🔍 DEBUG - Pre-creation Analysis:")
    logger.info(f"  Transformer config: num_layers={transformer_config.get('num_layers', 'unknown')}")
    logger.info(f"  Expected tensor count estimate: ~{transformer_config.get('num_layers', 40) * 20 + 50}")
    logger.info(f"  GGUF tensors available: {len(gguf_state_dict)}")

    try:
        from cinewan.modules.wan_transformer_gguf import create_wan_transformer_gguf

        # Create transformer with GGUF weights directly
        logger.info("\n🔄 Creating transformer and loading weights...")
        
        # CRITICAL FIX: Pre-process GGUF state dict to ensure norm layers are Parameters
        logger.info("🔧 Pre-processing GGUF tensors for norm layers...")
        processed_state_dict = {}
        norm_count = 0
        
        for key, tensor in gguf_state_dict.items():
            # Check if this is a norm layer
            if 'norm' in key.lower() or 'modulation' in key:
                # Convert to regular tensor if it's not already
                if hasattr(tensor, 'to_tensor'):
                    tensor = tensor.to_tensor()
                elif not isinstance(tensor, torch.Tensor):
                    logger.warning(f"  Unexpected type for {key}: {type(tensor)}")
                
                # Ensure it's float32 for norm layers
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(torch.float32)
                    norm_count += 1
                
                processed_state_dict[key] = tensor
            else:
                # Keep other tensors as-is (they can stay as GGMLTensor)
                processed_state_dict[key] = tensor
        
        logger.info(f"  Processed {norm_count} norm layer tensors")
        
        # Create transformer with processed weights
        gguf_transformer = create_wan_transformer_gguf(transformer_config, processed_state_dict)
        
        # Verify model integrity
        verify_model_integrity(gguf_transformer)

        # DEBUG: Post-creation analysis
        logger.info("\n📊 DEBUG - Post-creation Analysis:")

        # Check if transformer has expected structure - try both possible names
        blocks = getattr(gguf_transformer, 'blocks', getattr(gguf_transformer, 'transformer_blocks', None))
        if blocks is not None:
            num_blocks = len(blocks) if hasattr(blocks, '__len__') else 0
            logger.info(f"  Transformer blocks created: {num_blocks}")

            # Check first block structure with better error handling
            if num_blocks > 0:
                logger.info("  First block structure:")
                block = blocks[0]
                
                # Check various possible names for components
                component_names = {
                    'self_attn': ['self_attn', 'attn1', 'attn'],
                    'cross_attn': ['cross_attn', 'attn2'],
                    'ffn': ['ffn', 'ff', 'mlp'],
                    'norm1': ['norm1'],
                    'norm2': ['norm2'],
                    'norm3': ['norm3'],
                }
                
                for comp_type, possible_names in component_names.items():
                    found = False
                    for name in possible_names:
                        if hasattr(block, name):
                            module = getattr(block, name)
                            has_weight = hasattr(module, 'weight') if module else False
                            if has_weight:
                                weight = module.weight
                                weight_loaded = weight is not None
                                if weight_loaded:
                                    weight_type = type(weight).__name__
                                    weight_shape = weight.shape if hasattr(weight, 'shape') else 'unknown'
                                    logger.info(f"    {comp_type} (as {name}): ✅ loaded, type={weight_type}, shape={weight_shape}")
                                else:
                                    logger.info(f"    {comp_type} (as {name}): ❌ weight is None")
                            else:
                                logger.info(f"    {comp_type} (as {name}): exists but no weight attribute")
                            found = True
                            break
                    if not found:
                        logger.info(f"    {comp_type}: ❌ not found")

        # Count loaded parameters with better detection
        total_params = 0
        loaded_params = 0
        empty_params = []
        param_types = defaultdict(int)

        for name, param in gguf_transformer.named_parameters():
            if param is not None:
                param_type = type(param).__name__
                param_types[param_type] += 1
                
                if hasattr(param, 'numel'):
                    num_elements = param.numel()
                    if num_elements > 0:
                        total_params += num_elements
                        loaded_params += 1
                    else:
                        empty_params.append(name)
                else:
                    # Count it anyway if it exists
                    loaded_params += 1
            else:
                empty_params.append(name)

        logger.info(f"\n  Parameter Statistics:")
        logger.info(f"    Total parameter count: {loaded_params}")
        logger.info(f"    Total parameter elements: {total_params/1e9:.2f}B")
        logger.info(f"    Empty/None parameters: {len(empty_params)}")
        
        logger.info(f"\n  Parameter types found:")
        for ptype, count in param_types.items():
            logger.info(f"    {ptype}: {count}")

        if empty_params:
            logger.warning(f"\n  ⚠️ Empty parameters (first 10):")
            for param_name in empty_params[:10]:
                logger.warning(f"    - {param_name}")

        # Check critical components
        logger.info(f"\n  Critical Components Check:")
        critical = {
            'patch_embedding': hasattr(gguf_transformer, 'patch_embedding'),
            'time_embedding': hasattr(gguf_transformer, 'time_embedding'),
            'text_embedding': hasattr(gguf_transformer, 'text_embedding'),
            'head': hasattr(gguf_transformer, 'head'),
        }
        for name, exists in critical.items():
            if exists:
                module = getattr(gguf_transformer, name)
                has_weight = hasattr(module, 'weight') if module else False
                if has_weight:
                    weight = module.weight if hasattr(module, 'weight') else None
                    weight_loaded = weight is not None
                    if weight_loaded:
                        weight_type = type(weight).__name__
                        logger.info(f"    {name}: ✅ loaded, type={weight_type}")
                    else:
                        logger.info(f"    {name}: ❌ weight is None")
                else:
                    logger.info(f"    {name}: exists but no weight")
            else:
                logger.warning(f"    {name}: ❌ NOT FOUND")

        logger.info("\n✅ GGUF transformer created successfully!")

    except Exception as e:
        logger.error(f"Failed to create GGUF transformer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    logger.info(f"⏱️ Transformer created in {time.time() - start_transformer:.2f}s")

    # Create pipeline with loaded components
    start_pipeline = time.time()
    pipeline = pipeline_class(
        transformer=gguf_transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=pipeline_kwargs.get('scheduler', None),
    )
    logger.info(f"⏱️ Pipeline assembled in {time.time() - start_pipeline:.2f}s")

    # Don't move to device yet - let the memory mode handle it
    # pipeline.to(device)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("📋 FINAL LOADING SUMMARY:")
    logger.info(f"  GGUF file: {Path(gguf_path).name}")
    logger.info(f"  GGUF tensors: {len(gguf_state_dict)}")
    logger.info(f"  Total parameters in GGUF: {total_params/1e9:.2f}B")
    logger.info(f"  Transformer created: ✅")
    logger.info(f"  VAE loaded: ✅")
    logger.info(f"  Text Encoder loaded: ✅")
    logger.info(f"  Pipeline assembled: ✅")
    logger.info("="*60 + "\n")

    logger.info("✅ GGUF pipeline loaded successfully!")
    return pipeline


def load_gguf_transformer(
    gguf_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    dequant_dtype: Optional[torch.dtype] = None,
    patch_dtype: Optional[torch.dtype] = None,
) -> WanTransformer3DModel:
    """
    Load just the transformer from a GGUF file.

    Args:
        gguf_path: Path to the .gguf file
        config: Transformer configuration (will try to auto-detect if None)
        device: Device to load on
        dtype: Computation dtype
        dequant_dtype: Dequantization dtype
        patch_dtype: Patch/LoRA dtype

    Returns:
        WanTransformer3DModel instance with GGUF weights
    """
    # Initialize custom operations
    ops = GGMLOps()
    if dequant_dtype is not None:
        ops.Linear.dequant_dtype = dequant_dtype
    if patch_dtype is not None:
        ops.Linear.patch_dtype = patch_dtype

    # Load GGUF state dict
    gguf_state_dict = gguf_sd_loader(gguf_path)

    if config is None:
        # Default config for 14B model with all required fields
        config = {
            "dim": 5120,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "in_channels": 16,
            "out_dim": 16,
            "out_channels": 16,
            "model_type": "t2v",
            "num_heads": 40,
            "num_layers": 40,
            "text_len": 512,
            "text_dim": 4096,
            "patch_size": (1, 2, 2),
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
        }
    
    # Pre-process GGUF state dict for norm layers
    logger.info("🔧 Pre-processing GGUF tensors...")
    processed_state_dict = {}
    
    for key, tensor in gguf_state_dict.items():
        if 'norm' in key.lower() or 'modulation' in key:
            if hasattr(tensor, 'to_tensor'):
                tensor = tensor.to_tensor()
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(torch.float32)
            processed_state_dict[key] = tensor
        else:
            processed_state_dict[key] = tensor

    # Import and create transformer
    from cinewan.modules.wan_transformer_gguf import create_wan_transformer_gguf
    transformer = create_wan_transformer_gguf(config, processed_state_dict)
    
    # Verify integrity
    verify_model_integrity(transformer)

    # Note: Don't move to device here if using cpu offload
    # transformer.to(device=device, dtype=dtype)

    return transformer
