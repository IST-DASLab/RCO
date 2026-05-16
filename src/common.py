"""Common utilities for model manipulation and tensor operations."""

import random
import dataclasses
from typing import Any, Sequence, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn


def get_input_device(model: nn.Module) -> torch.device:
    """Resolve the device where the model expects its input tensors.

    For models sharded with device_map='auto' this returns the device of the
    input embeddings (typically cuda:0). For single-GPU models it returns the
    device the model is on. Falls back to the first parameter when neither is
    available.
    """
    try:
        return next(model.get_input_embeddings().parameters()).device
    except (StopIteration, AttributeError):
        return next(model.parameters()).device


def resolve_module(model: nn.Module, dotted_path: str) -> nn.Module:
    """Walk a dotted attribute / index path under model. Used for layer lookup.

    Integer-looking segments use __getitem__ (for nn.ModuleList / nn.Sequential);
    everything else uses getattr.
    """
    current = model
    for part in dotted_path.split('.'):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def fix_seed(seed: int):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to(data: Any, *args, **kwargs):
    """
    Recursively move tensors, dataclasses, and nested structures to a device/dtype.
    
    Handles tensors, tuples, lists, sets, dicts, and dataclasses.
    Non-tensor values pass through unchanged.
    """
    def _to(x):
        return to(x, *args, **kwargs)

    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(_to(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, _to(v)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: _to(v) for k, v in vars(data).items()})
    else:
        return data


def maybe_first_element(x):
    """Extract first element if x is a sequence, otherwise return x."""
    if isinstance(x, Sequence):
        x = x[0]
    return x


def get_layer_weights(model: nn.Module, layer_name: str) -> Optional[torch.Tensor]:
    """Return the .weight tensor at a dotted-path layer, or None if missing.

    None is returned when the path does not resolve, the resolved module has
    no .weight attribute, or the weight is on the meta device.
    """
    try:
        current = resolve_module(model, layer_name)
    except (AttributeError, IndexError, KeyError):
        return None
    if not hasattr(current, 'weight'):
        return None
    weight = current.weight.detach()
    if weight.device.type == 'meta':
        return None
    return weight


def set_layer_weights(model: nn.Module, layer_name: str, weights: torch.Tensor):
    """Write into the .weight tensor at a dotted-path layer.

    Casts to the target layer's device and dtype first.
    """
    final_module = resolve_module(model, layer_name)
    final_module.weight.data = weights.to(
        device=final_module.weight.device, 
        dtype=final_module.weight.dtype
    )


def get_layer_param_counts(model: nn.Module, layer_names: List[str]) -> Dict[str, int]:
    """Get parameter counts for each layer."""
    param_counts = {}
    for name in layer_names:
        weights = get_layer_weights(model, name)
        if weights is not None:
            param_counts[name] = weights.numel()
        else:
            param_counts[name] = 1
    return param_counts


def compute_layer_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """
    Compute relative squared Frobenius error between original and quantized weights.
    Returns ||Q - W||_F^2 / ||W||_F^2
    """
    with torch.no_grad():
        # Move both tensors to CPU to avoid device mismatch
        original = original.cpu()
        quantized = quantized.cpu()
        
        return ((torch.norm(quantized - original, p='fro') ** 2) / 
                (torch.norm(original, p='fro') ** 2)).item()


def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
