"""
Persisted quantization parameters per (layer, bitwidth).

run_quantize.py writes one of these alongside the dequantized fake-quant tensor:

    layer_dir/
      model.layers.0.self_attn.q_proj/
        4.pth              # dequantized FP/BF tensor (used by search pipeline)
        4_qparams.pt       # this file: integer codes + scales + zeros + perm + meta
        5.pth
        5_qparams.pt
        ...

Downstream checkpoint packers (vLLM compressed-tensors, Humming, GPTQModel, ...)
load the qparams bundle directly and pack into their target on-disk format. No
re-quantization of the dequantized tensor is needed.

Layout of a saved bundle (a single torch-pickled dict):

    qweight     uint8 [d_row, d_col]            integer codes in [0, 2**bits - 1].
                                                Stored in the column order produced
                                                by GPTQ; if act_order was used
                                                this is the act-order-permuted
                                                column order, see perm.
    scales      original-dtype [d_row, n_groups]   per-(row, group) scale.
    zeros       original-dtype [d_row, n_groups]   per-(row, group) zero-point.
                                                Asymmetric quant: real-valued.
                                                Symmetric quant: zero-tensor.
    perm        int64 [d_col] | None              act-order column permutation that
                                                was applied to the input weights
                                                before quantization.  None when
                                                act_order=False.
    bits        int                              4..8.
    group_size  int                              group size along input dim.
    sym         bool                             symmetric grid?
    perchannel  bool                             per-channel scales?
    act_order   bool                             was activation-order GPTQ used?
    shape       tuple[int, int]                  original (out_features, in_features).
    dtype       str                              original weight dtype, e.g. "bfloat16".
    schema      int                              format version.

Reconstruction of the dequantized weight (column-permuted form):

    W_q = scales[:, group_idx] * (qweight - zeros[:, group_idx])

where group_idx[c] = c // group_size. To recover the *original* column order,
apply inverse_perm = torch.argsort(perm) to the columns of W_q. Most fast
inference kernels (e.g. Marlin) prefer to keep the permuted layout and permute
activations at runtime instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

QPARAMS_SCHEMA_VERSION = 1
QPARAMS_SUFFIX = "_qparams.pt"


def qparams_path(layer_dir: str | os.PathLike, bits: int) -> str:
    """Canonical path of the qparams bundle for one (layer, bitwidth)."""
    return os.path.join(str(layer_dir), f"{int(bits)}{QPARAMS_SUFFIX}")


def save_qparams(
    layer_dir: str | os.PathLike,
    *,
    bits: int,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    perm: Optional[torch.Tensor],
    handle: Any,
) -> str:
    """
    Persist a single (layer, bitwidth) qparams bundle. handle is the FastOBQ
    instance; we read group_size / sym / perchannel / act_order / shape / dtype
    off it so the call site stays tight.
    """
    quantizer = handle.quantizer_dict[bits]
    bundle: Dict[str, Any] = {
        "qweight": qweight.detach().cpu().contiguous(),
        "scales":  scales.detach().cpu().contiguous(),
        "zeros":   zeros.detach().cpu().contiguous(),
        "perm":    None if perm is None else perm.detach().cpu().contiguous(),
        "bits":       int(bits),
        "group_size": int(handle.group_size if handle.group_size else handle.d_col),
        "sym":        bool(quantizer.sym),
        "perchannel": bool(quantizer.perchannel),
        "act_order":  bool(handle.act_order),
        "shape":      tuple(int(x) for x in handle.W_shape),
        "dtype":      str(handle.W_dtype).replace("torch.", ""),
        "schema":     QPARAMS_SCHEMA_VERSION,
    }
    path = qparams_path(layer_dir, bits)
    torch.save(bundle, path)
    return path


def load_qparams(layer_dir: str | os.PathLike, bits: int) -> Dict[str, Any]:
    """Load a qparams bundle. Raises FileNotFoundError if absent."""
    path = qparams_path(layer_dir, bits)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    schema = bundle.get("schema", 0)
    if schema != QPARAMS_SCHEMA_VERSION:
        raise ValueError(
            f"qparams schema mismatch at {path}: got {schema}, expected {QPARAMS_SCHEMA_VERSION}"
        )
    return bundle


def dequantize_from_qparams(bundle: Dict[str, Any]) -> torch.Tensor:
    """
    Reconstruct the dequantized weight tensor from a qparams bundle, in the
    *column-permuted* order used during quantization. Apply
    tensor[:, torch.argsort(bundle['perm'])] afterwards if you want the
    original column order.
    """
    qweight = bundle["qweight"].to(torch.float32)
    scales  = bundle["scales"].to(torch.float32)
    zeros   = bundle["zeros"].to(torch.float32)
    group_size = bundle["group_size"]
    d_row, d_col = qweight.shape

    # Map each column to its group index, then broadcast to (d_row, d_col).
    group_idx = torch.arange(d_col, device=qweight.device) // group_size
    s = scales[:, group_idx]
    z = zeros[:, group_idx]
    w = s * (qweight - z)

    target_dtype = getattr(torch, bundle["dtype"])
    return w.to(target_dtype)
