# cinewan/modules/wan_transformer_gguf.py
# GGUF-optimized version of WanTransformer3DModel
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
# This file is a modified version of:
#   https://github.com/aigc-apps/VideoX-Fun/blob/main/videox_fun/models/wan_transformer3d.py
# which was originally based on:
#   https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
#
# GGUF support and further modifications by https://github.com/acaimari
#
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

# Import GGML operations from our loader
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cinewan.gguf_loader.ops import GGMLOps

# Import necessary components from VideoX-Fun
from videox_fun.models.wan_transformer3d import (
    WanRMSNorm, WanLayerNorm, rope_params, rope_apply_qk,
    attention, sinusoidal_embedding_1d
)


class WanTransformerGGUF(ModelMixin, ConfigMixin):
    """
    GGUF-optimized Wan Transformer for video generation.
    This version creates layers with GGML operations from the start to avoid memory issues.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        in_channels=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        out_channels=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # GGUF specific parameters
        use_ggml_ops=True,
        dequant_dtype=None,
        patch_dtype=None,
    ):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Initialize GGML operations if requested
        if use_ggml_ops:
            self.ops = GGMLOps()
            if dequant_dtype is not None:
                self.ops.Linear.dequant_dtype = dequant_dtype
            if patch_dtype is not None:
                self.ops.Linear.patch_dtype = patch_dtype
        else:
            self.ops = nn

        # Create layers with GGML operations
        self._create_layers()

        # Initialize buffers
        self._init_buffers()

        # Additional attributes for compatibility
        self.gradient_checkpointing = False
        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def _create_layers(self):
        """Create all layers using GGML operations"""

        # Embeddings
        self.patch_embedding = self.ops.Conv3d(
            self.in_dim, self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Text embedding uses sequential layers
        self.text_embedding = nn.Sequential(
            self.ops.Linear(self.text_dim, self.dim),
            nn.GELU(approximate='tanh'),
            self.ops.Linear(self.dim, self.dim)
        )

        # Time embeddings
        self.time_embedding = nn.Sequential(
            self.ops.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            self.ops.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            self.ops.Linear(self.dim, self.dim * 6)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlockGGUF(
                self.ops,
                't2v_cross_attn' if self.model_type == 't2v' else 'i2v_cross_attn',
                self.dim, self.ffn_dim, self.num_heads,
                self.window_size, self.qk_norm, self.cross_attn_norm, self.eps
            )
            for _ in range(self.num_layers)
        ])

        # Output head
        self.head = HeadGGUF(self.ops, self.dim, self.out_dim, self.patch_size, self.eps)

        # Image embedding for i2v
        if self.model_type == 'i2v':
            self.img_emb = MLPProjGGUF(self.ops, 1280, self.dim)

    def _init_buffers(self):
        """Initialize rope frequencies"""
        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads
        self.d = d

        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        **kwargs
    ):
        """Forward pass compatible with original WanTransformer3DModel"""

        # Handle device
        device = x[0].device if isinstance(x, list) else x.device
        dtype = x[0].dtype if isinstance(x, list) else x.dtype

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Combine conditionals if needed
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        # Grid sizes
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )

        # Flatten spatial dimensions
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        # Pad sequences
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # Time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float()
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # Text embeddings
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        # Add image embeddings for i2v
        if self.model_type == 'i2v' and clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(
                x, e0, seq_lens, grid_sizes, self.freqs,
                context, None, dtype, t
            )

        # Output head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        return x

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video from patches"""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def _set_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    # Compatibility methods
    def enable_teacache(self, *args, **kwargs):
        pass
    def enable_cfg_skip(self, *args, **kwargs):
        pass
    def enable_riflex(self, *args, **kwargs):
        pass


class WanAttentionBlockGGUF(nn.Module):
    """GGUF-optimized attention block"""

    def __init__(
        self,
        ops,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.ops = ops
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # CRITICAL FIX: norm1 and norm2 don't exist in GGUF - they use AdaLayerNorm
        # Create them as LayerNorm WITHOUT learnable parameters
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # Attention layers with GGML ops
        self.self_attn = WanSelfAttentionGGUF(ops, dim, num_heads, window_size, qk_norm, eps)
        self.cross_attn = WanCrossAttentionGGUF(ops, cross_attn_type, dim, num_heads, qk_norm, eps)
        
        # FFN with GGML ops
        self.ffn = nn.Sequential(
            ops.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            ops.Linear(ffn_dim, dim)
        )

        # Modulation parameter
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dtype, t):
        e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t)
        x = x + y * e[2]

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t)

        # FFN
        temp_x = self.norm2(x) * (1 + e[4]) + e[3]
        temp_x = temp_x.to(dtype)
        y = self.ffn(temp_x)
        x = x + y * e[5]

        return x


class WanSelfAttentionGGUF(nn.Module):
    """GGUF-optimized self-attention"""

    def __init__(self, ops, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.ops = ops
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Linear layers with GGML ops
        self.q = ops.Linear(dim, dim)
        self.k = ops.Linear(dim, dim)
        self.v = ops.Linear(dim, dim)
        self.o = ops.Linear(dim, dim)
        
        # CRITICAL: Use regular RMSNorm, not GGML ops!
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype, t):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # QKV computation
        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)

        # Apply RoPE
        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        # Attention
        x = attention(
            q.to(dtype), k.to(dtype), v=v.to(dtype),
            k_lens=seq_lens, window_size=self.window_size
        )
        x = x.to(dtype)

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttentionGGUF(nn.Module):
    """GGUF-optimized cross-attention"""

    def __init__(self, ops, cross_attn_type, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.ops = ops
        self.cross_attn_type = cross_attn_type
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # Linear layers with GGML ops
        self.q = ops.Linear(dim, dim)
        self.k = ops.Linear(dim, dim)
        self.v = ops.Linear(dim, dim)
        self.o = ops.Linear(dim, dim)
        
        # CRITICAL: Use regular RMSNorm, not GGML ops!
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Additional layers for i2v
        if cross_attn_type == 'i2v_cross_attn':
            self.k_img = ops.Linear(dim, dim)
            self.v_img = ops.Linear(dim, dim)
            self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype, t):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        if self.cross_attn_type == 'i2v_cross_attn' and context.size(1) > 257:
            # Handle i2v case
            context_img = context[:, :257]
            context = context[:, 257:]

            q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
            k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
            v = self.v(context.to(dtype)).view(b, -1, n, d)
            k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
            v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

            # Image attention
            img_x = attention(q.to(dtype), k_img.to(dtype), v_img.to(dtype), k_lens=None)
            img_x = img_x.to(dtype)

            # Text attention
            x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
            x = x.to(dtype)

            # Combine
            x = x.flatten(2)
            img_x = img_x.flatten(2)
            x = x + img_x
        else:
            # Standard cross-attention
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

            x = attention(q, k, v, k_lens=context_lens)
            x = x.flatten(2)

        x = self.o(x)
        return x


class HeadGGUF(nn.Module):
    """GGUF-optimized output head"""

    def __init__(self, ops, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.ops = ops
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Layers
        out_dim = math.prod(patch_size) * out_dim
        # CRITICAL: Use regular LayerNorm, not GGML ops!
        self.norm = WanLayerNorm(dim, eps)
        self.head = ops.Linear(dim, out_dim)

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProjGGUF(nn.Module):
    """GGUF-optimized MLP projection for image embeddings"""

    def __init__(self, ops, in_dim, out_dim):
        super().__init__()
        self.ops = ops
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            ops.Linear(in_dim, in_dim),
            nn.GELU(),
            ops.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


def create_wan_transformer_gguf(config, state_dict):
    """
    Create a WanTransformerGGUF and load GGUF weights.
    VERSIÓN CORREGIDA con mapeo mejorado de norm layers
    """
    import torch.nn as nn

    # Asegurar configuración completa
    if 'in_channels' not in config:
        config['in_channels'] = config.get('in_dim', 16)
    if 'out_channels' not in config:
        config['out_channels'] = config.get('out_dim', 16)
    if 'text_dim' not in config:
        config['text_dim'] = 4096

    print("\n📋 Creating GGUF transformer structure...")
    model = WanTransformerGGUF(**config)

    print("\n🔄 Assigning GGUF weights directly to GGML layers...")

    loaded_keys = []
    failed_keys = []
    
    # Función auxiliar para asignar peso
    def assign_weight(module, param_name, tensor):
        """Asigna un tensor a un parámetro del módulo"""
        if hasattr(tensor, 'to_tensor'):
            # Es un GGMLTensor, mantenerlo como está para layers GGML
            setattr(module, param_name, tensor)
        elif isinstance(tensor, torch.Tensor):
            # Para norm layers, convertir a Parameter
            if 'norm' in module.__class__.__name__.lower():
                setattr(module, param_name, nn.Parameter(tensor.to(torch.float32)))
            else:
                setattr(module, param_name, tensor)
        else:
            setattr(module, param_name, tensor)

    # 1. Cargar embeddings y componentes principales
    critical_mappings = [
        ('patch_embedding', ['weight', 'bias']),
        ('text_embedding.0', ['weight', 'bias']),
        ('text_embedding.2', ['weight', 'bias']),
        ('time_embedding.0', ['weight', 'bias']),
        ('time_embedding.2', ['weight', 'bias']),
        ('time_projection.1', ['weight', 'bias']),
        ('head.head', ['weight', 'bias']),
        ('head.norm', ['weight', 'bias']),
    ]

    for layer_name, params in critical_mappings:
        for param_name in params:
            full_key = f"{layer_name}.{param_name}"
            if full_key in state_dict:
                try:
                    parts = layer_name.split('.')
                    module = model
                    for part in parts:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    
                    assign_weight(module, param_name, state_dict[full_key])
                    loaded_keys.append(full_key)
                    print(f"  ✅ Loaded {full_key}")
                except Exception as e:
                    failed_keys.append((full_key, str(e)))
                    print(f"  ❌ Failed {full_key}: {e}")

    # 2. Cargar bloques del transformer con mapeo mejorado
    print("\n🔄 Loading transformer block weights with improved mapping...")
    
    for key, tensor in state_dict.items():
        if not key.startswith('blocks.'):
            continue
            
        if key in loaded_keys:
            continue
            
        try:
            parts = key.split('.')
            block_idx = int(parts[1])
            
            if block_idx >= len(model.blocks):
                continue
                
            block = model.blocks[block_idx]
            remaining_path = '.'.join(parts[2:])
            
            # Mapeo específico para cada componente
            if remaining_path == 'modulation':
                # Modulation es un Parameter directo
                if isinstance(tensor, torch.Tensor):
                    block.modulation = nn.Parameter(tensor.to(torch.float32))
                else:
                    block.modulation = tensor
                loaded_keys.append(key)
                
            elif remaining_path.startswith('norm'):
                # norm1, norm2, norm3
                norm_name = remaining_path.split('.')[0]
                param_type = remaining_path.split('.')[1] if '.' in remaining_path else 'weight'
                
                if hasattr(block, norm_name):
                    norm_layer = getattr(block, norm_name)
                    if isinstance(tensor, torch.Tensor):
                        setattr(norm_layer, param_type, nn.Parameter(tensor.to(torch.float32)))
                    else:
                        # Si es GGMLTensor, convertir a tensor normal para norm layers
                        if hasattr(tensor, 'to_tensor'):
                            setattr(norm_layer, param_type, nn.Parameter(tensor.to_tensor().to(torch.float32)))
                        else:
                            setattr(norm_layer, param_type, tensor)
                    loaded_keys.append(key)
                    
            elif remaining_path.startswith('self_attn.'):
                # Self attention
                attn_path = remaining_path.replace('self_attn.', '')
                
                if attn_path.startswith('norm'):
                    # norm_q, norm_k dentro de self_attn
                    norm_name = attn_path.split('.')[0]
                    if hasattr(block.self_attn, norm_name):
                        norm_layer = getattr(block.self_attn, norm_name)
                        if isinstance(tensor, torch.Tensor):
                            norm_layer.weight = nn.Parameter(tensor.to(torch.float32))
                        elif hasattr(tensor, 'to_tensor'):
                            norm_layer.weight = nn.Parameter(tensor.to_tensor().to(torch.float32))
                        loaded_keys.append(key)
                else:
                    # q, k, v, o weights
                    param_parts = attn_path.split('.')
                    if len(param_parts) == 2:
                        layer_name, param_type = param_parts
                        if hasattr(block.self_attn, layer_name):
                            layer = getattr(block.self_attn, layer_name)
                            assign_weight(layer, param_type, tensor)
                            loaded_keys.append(key)
                            
            elif remaining_path.startswith('cross_attn.'):
                # Cross attention
                attn_path = remaining_path.replace('cross_attn.', '')
                
                if attn_path.startswith('norm'):
                    # norm_q, norm_k dentro de cross_attn
                    norm_name = attn_path.split('.')[0]
                    if hasattr(block.cross_attn, norm_name):
                        norm_layer = getattr(block.cross_attn, norm_name)
                        if isinstance(tensor, torch.Tensor):
                            norm_layer.weight = nn.Parameter(tensor.to(torch.float32))
                        elif hasattr(tensor, 'to_tensor'):
                            norm_layer.weight = nn.Parameter(tensor.to_tensor().to(torch.float32))
                        loaded_keys.append(key)
                else:
                    # q, k, v, o weights
                    param_parts = attn_path.split('.')
                    if len(param_parts) == 2:
                        layer_name, param_type = param_parts
                        if hasattr(block.cross_attn, layer_name):
                            layer = getattr(block.cross_attn, layer_name)
                            assign_weight(layer, param_type, tensor)
                            loaded_keys.append(key)
                            
            elif remaining_path.startswith('ffn.'):
                # FFN layers
                ffn_path = remaining_path.replace('ffn.', '')
                if ffn_path in ['0.weight', '0.bias', '2.weight', '2.bias']:
                    ffn_idx = int(ffn_path.split('.')[0])
                    param_type = ffn_path.split('.')[1]
                    if ffn_idx < len(block.ffn):
                        assign_weight(block.ffn[ffn_idx], param_type, tensor)
                        loaded_keys.append(key)
                        
        except Exception as e:
            failed_keys.append((key, str(e)))

    print(f"\n📊 Block loading statistics:")
    print(f"  ✅ Weights loaded: {len(loaded_keys)}")
    print(f"  ❌ Weights failed: {len(failed_keys)}")

    if failed_keys:
        print("\n🔍 Sample of failed weights:")
        for key, error in failed_keys[:5]:
            print(f"    {key}: {error[:100]}")

    # 3. Verificación final
    print("\n🔍 Final verification of critical layers:")
    
    # Verificar bloques
    for i in range(min(3, len(model.blocks))):
        block = model.blocks[i]
        print(f"\n  Block {i}:")
        
        # Verificar norm layers - norm1 y norm2 no tienen weights (AdaLayerNorm)
        for norm_name in ['norm1', 'norm2', 'norm3']:
            if hasattr(block, norm_name):
                norm = getattr(block, norm_name)
                if norm_name in ['norm1', 'norm2']:
                    # These are LayerNorm without affine parameters
                    print(f"    {norm_name}: ✅ AdaLayerNorm (no weights needed)")
                elif hasattr(norm, 'weight') and norm.weight is not None:
                    print(f"    {norm_name}: ✅ shape={norm.weight.shape}")
                else:
                    print(f"    {norm_name}: ❌ No weight")
        
        # Verificar modulation
        if hasattr(block, 'modulation'):
            if block.modulation is not None:
                print(f"    modulation: ✅ shape={block.modulation.shape}")
        
        # Verificar attention
        if hasattr(block.self_attn, 'q') and block.self_attn.q.weight is not None:
            print(f"    self_attn.q: ✅")
        if hasattr(block.cross_attn, 'q') and block.cross_attn.q.weight is not None:
            print(f"    cross_attn.q: ✅")
        if len(block.ffn) > 0 and hasattr(block.ffn[0], 'weight') and block.ffn[0].weight is not None:
            print(f"    ffn[0]: ✅")

    print(f"\n✅ Total loaded: {len(loaded_keys)}/{len(state_dict)} tensors")
    
    # CRITICAL: Initialize norm1 and norm2 as they don't have weights in GGUF
    print("\n🔧 Post-processing: AdaLayerNorm setup...")
    for i, block in enumerate(model.blocks):
        # norm1 and norm2 are LayerNorm without affine - they're modulated by 'modulation'
        # This is the AdaLayerNorm pattern used in DiT/Wan models
        if hasattr(block, 'norm1'):
            block.norm1.eval()  # Always in eval mode to use running stats
        if hasattr(block, 'norm2'):
            block.norm2.eval()  # Always in eval mode to use running stats
    
    print("✅ AdaLayerNorm initialization complete")
    
    return model
