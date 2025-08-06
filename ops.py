# gguf_loader/ops.py
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

import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple, Any

from .dequant import dequantize_tensor, is_quantized


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights.
    """
    def __init__(self, *args, tensor_type=None, tensor_shape=None, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type if tensor_type is not None else gguf.GGMLQuantizationType.F32
        self.tensor_shape = tensor_shape if tensor_shape is not None else self.shape
        self.patches = patches if patches is not None else []

    def __new__(cls, *args, tensor_type=None, tensor_shape=None, patches=None, **kwargs):
        # Asegurar que los parámetros requeridos estén presentes
        if tensor_type is None:
            tensor_type = gguf.GGMLQuantizationType.F32
        if tensor_shape is None and len(args) > 0:
            # Intentar obtener shape del primer argumento si es un tensor
            if isinstance(args[0], torch.Tensor):
                tensor_shape = args[0].shape
            else:
                tensor_shape = torch.Size([])
        
        instance = super().__new__(cls, *args, **kwargs)
        instance.tensor_type = tensor_type
        instance.tensor_shape = tensor_shape
        instance.patches = patches if patches is not None else []
        return instance

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", gguf.GGMLQuantizationType.F32)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        new = super().clone(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", gguf.GGMLQuantizationType.F32)
        new.tensor_shape = getattr(self, "tensor_shape", self.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def detach(self, *args, **kwargs):
        new = super().detach(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", gguf.GGMLQuantizationType.F32)
        new.tensor_shape = getattr(self, "tensor_shape", self.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Ignoring 'copy_' on tensor: {e}")
            return self

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = super().shape
        return self.tensor_shape


class GGMLLayer(nn.Module):
    """
    Base class for layers that de-quantize weights on the fly.
    """
    dequant_dtype = None
    patch_dtype = None

    def __init__(self):
        # Only call nn.Module.__init__ here, not super()
        nn.Module.__init__(self)
        # Explicitly initialize weight and bias as None
        self.weight = None
        self.bias = None

    def is_ggml_quantized(self, weight=None, bias=None):
        if weight is None:
            weight = self.weight if hasattr(self, 'weight') else None
        if bias is None:
            bias = self.bias if hasattr(self, 'bias') else None
        
        # Verificar si es un GGMLTensor o está cuantizado
        return (weight is not None and hasattr(weight, 'tensor_type')) or \
               (bias is not None and hasattr(bias, 'tensor_type')) or \
               is_quantized(weight) or is_quantized(bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return None

        # Dequantize tensor
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # Prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if device is None:
                device = input.device

        bias = None
        if hasattr(self, 'bias') and self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = bias.to(dtype=dtype, device=device)

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = weight.to(dtype=dtype, device=device)

        return weight, bias


class GGMLOps:
    """
    Custom operations that dequantize weights on the fly.
    """

    class Linear(GGMLLayer):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # Initialize GGMLLayer first (which calls nn.Module.__init__)
            super().__init__()
            # Store the parameters we need
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            # Weight and bias are already None from GGMLLayer.__init__

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return F.linear(input, weight, bias)
            else:
                # Asegurar tipos compatibles
                weight = self.weight
                bias = self.bias
                if weight is not None:
                    weight = weight.to(dtype=input.dtype, device=input.device)
                if bias is not None:
                    bias = bias.to(dtype=input.dtype, device=input.device)
                return F.linear(input, weight, bias)
        
        def extra_repr(self):
            return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'

    class Conv2d(GGMLLayer):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                     device=None, dtype=None):
            super().__init__()
            # Store Conv2d parameters
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            self.groups = groups
            self.padding_mode = padding_mode
            self.use_bias = bias
            # Calculate padding for _conv_forward
            self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        def _conv_forward(self, input, weight, bias):
            # Asegurar que weight y bias tienen el mismo dtype que input
            if weight is not None:
                weight = weight.to(dtype=input.dtype, device=input.device)
            if bias is not None:
                bias = bias.to(dtype=input.dtype, device=input.device)
                
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride, (0, 0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return self._conv_forward(input, weight, bias)
            else:
                return self._conv_forward(input, self.weight, self.bias)

    class Conv3d(GGMLLayer):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                     device=None, dtype=None):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            # Ensure kernel_size is a 3-tuple
            if isinstance(kernel_size, int):
                self.kernel_size = (kernel_size, kernel_size, kernel_size)
            elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
                self.kernel_size = (1, kernel_size[0], kernel_size[1])
            else:
                self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
            
            # Handle stride
            if isinstance(stride, int):
                self.stride = (stride, stride, stride)
            elif isinstance(stride, tuple) and len(stride) == 2:
                self.stride = (1, stride[0], stride[1])
            else:
                self.stride = stride if isinstance(stride, tuple) else (stride,) * 3

            # Handle padding
            if isinstance(padding, int):
                self.padding = (padding, padding, padding)
            elif isinstance(padding, tuple) and len(padding) == 2:
                self.padding = (0, padding[0], padding[1])
            else:
                self.padding = padding if isinstance(padding, tuple) else (padding,) * 3

            # Handle dilation
            if isinstance(dilation, int):
                self.dilation = (dilation, dilation, dilation)
            elif isinstance(dilation, tuple) and len(dilation) == 2:
                self.dilation = (1, dilation[0], dilation[1])
            else:
                self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
            
            self.groups = groups
            self.padding_mode = padding_mode
            self.use_bias = bias
            self.transposed = False
            self.output_padding = (0, 0, 0)
            # Calculate padding for _conv_forward
            self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        def _conv_forward(self, input, weight, bias):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride, (0, 0, 0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        def forward(self, input):
            # Siempre obtener weight y bias con el tipo correcto
            weight = self.weight
            bias = self.bias
            
            # Si hay tensores GGML, usar cast_bias_weight para dequantizar
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
            
            # Si weight/bias existen, asegurar que tienen el tipo correcto
            if weight is not None:
                weight = weight.to(dtype=input.dtype, device=input.device)
            if bias is not None:
                bias = bias.to(dtype=input.dtype, device=input.device)
            
            # Hacer la convolución
            if weight is not None:
                return self._conv_forward(input, weight, bias)
            else:
                # Sin pesos - esto causará error pero es mejor que silencioso
                raise RuntimeError("Conv3d weight is None!")

    class GroupNorm(GGMLLayer):
        def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                     device=None, dtype=None):
            super().__init__()
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            if not affine:
                self.weight = None
                self.bias = None

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return F.group_norm(
                    input, self.num_groups, weight, bias, self.eps
                )
            else:
                return F.group_norm(
                    input, self.num_groups, self.weight, self.bias, self.eps
                )

    class LayerNorm(GGMLLayer):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                     bias=True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.use_bias = bias
            if not elementwise_affine:
                self.weight = None
                self.bias = None

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, bias = self.cast_bias_weight(input)
                return F.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps
                )
            else:
                return F.layer_norm(
                    input, self.normalized_shape, self.weight, self.bias, self.eps
                )

    class Embedding(GGMLLayer):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                     max_norm=None, norm_type=2., scale_grad_by_freq=False,
                     sparse=False, device=None, dtype=None):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse

        def forward(self, input):
            if self.is_ggml_quantized():
                weight, _ = self.cast_bias_weight(input)
                return F.embedding(
                    input, weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
            else:
                return F.embedding(
                    input, self.weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
