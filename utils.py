# gguf_loader/utils.py
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
# Further modifications and PyTorch adaptation by Caimari:
#   https://github.com/acaimari
#
# Copyright (c) 2024 City96, Caimari. All rights reserved.

import torch
import logging
from typing import Dict, Any, Optional, List, Union


def move_patch_to_device(item, device):
    """
    Recursively move patches to specified device.
    """
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item


def apply_lora_patches(weight: torch.Tensor, patches: List, key: str, dtype: Optional[torch.dtype] = None):
    """
    Apply LoRA patches to a weight tensor.
    
    Args:
        weight: The base weight tensor
        patches: List of patches to apply
        key: Key identifying the weight
        dtype: Optional dtype for computation
    
    Returns:
        Weight tensor with patches applied
    """
    if not patches:
        return weight
    
    # This is a simplified version - you'd need to implement
    # the actual LoRA math based on your specific requirements
    for patch in patches:
        # Apply patch logic here
        pass
    
    return weight


def estimate_memory_usage(state_dict: Dict[str, Any], dtype: torch.dtype = torch.float16) -> int:
    """
    Estimate memory usage for a state dict.
    
    Args:
        state_dict: Model state dict
        dtype: Target dtype for dequantization
    
    Returns:
        Estimated memory usage in bytes
    """
    total_bytes = 0
    
    for key, tensor in state_dict.items():
        if hasattr(tensor, "tensor_shape"):
            shape = tensor.tensor_shape
        else:
            shape = tensor.shape
        
        # Calculate bytes based on dtype
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes += shape.numel() * element_size
    
    return total_bytes


def get_model_info(gguf_path: str) -> Dict[str, Any]:
    """
    Get information about a GGUF model without fully loading it.
    
    Args:
        gguf_path: Path to GGUF file
    
    Returns:
        Dictionary with model information
    """
    import gguf
    
    reader = gguf.GGUFReader(gguf_path)
    
    info = {
        "path": gguf_path,
        "tensors": len(reader.tensors),
        "architecture": None,
        "quantization_types": {},
        "total_parameters": 0,
    }
    
    # Get architecture
    arch_field = reader.get_field("general.architecture")
    if arch_field:
        info["architecture"] = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
    
    # Count parameters and quantization types
    for tensor in reader.tensors:
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        info["quantization_types"][tensor_type_str] = info["quantization_types"].get(tensor_type_str, 0) + 1
        
        # Estimate parameter count
        shape = list(reversed(tensor.shape))
        params = 1
        for dim in shape:
            params *= dim
        info["total_parameters"] += params
    
    return info


def print_memory_usage():
    """
    Print current GPU memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
