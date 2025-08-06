# gguf_loader/loader.py
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

import warnings
import logging
import torch
import gguf

from .ops import GGMLTensor
from .dequant import is_quantized

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image"}

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")

def gguf_sd_loader(path, handle_prefix=None, return_arch=False):
    """
    Read state dict as fake tensors from GGUF file.
    
    Args:
        path: Path to GGUF file
        handle_prefix: Optional prefix to strip from tensor names
        return_arch: Whether to return architecture string
    
    Returns:
        State dict with GGMLTensor objects (and optionally architecture string)
    """
    reader = gguf.GGUFReader(path)

    # Filter and strip prefix if needed
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix and handle_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # Detect and verify architecture
    arch_str = get_field(reader, "general.architecture", str)
    if arch_str not in IMG_ARCH_LIST:
        logging.warning(f"Unknown architecture in GGUF file: {arch_str}")
        # For Wan models, we'll assume it's compatible
        if "wan" not in str(path).lower() and arch_str != "wan":
            logging.warning("This might not be a Wan model GGUF file")

    # Main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        
        # Avoid numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # Add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # Keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # Print loaded tensor type counts
    logging.info("GGUF qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # Mark largest tensor for VRAM estimation
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    if return_arch:
        return (state_dict, arch_str)
    return state_dict
