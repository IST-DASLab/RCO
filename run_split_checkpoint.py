#!/usr/bin/env python3
"""Split a quantized HuggingFace checkpoint into the per-layer database.

Reads a uniformly-quantized HF model in the compressed-tensors format
(see https://github.com/neuralmagic/compressed-tensors) and writes two
files per linear layer:

  <layer-dir>/<layer_name>/<bw>.pth           dequantized fake-quant tensor
                                              (consumed by the search)
  <layer-dir>/<layer_name>/<bw>_qparams.pt    integer codes + scales +
                                              zeros + meta
                                              (consumed by checkpoint packers)

This populates the chosen bitwidth slot of the per-layer database that
the search scripts and run_build_checkpoint.py read. Other bitwidths
must still be filled in by run_quantize.py before the search has
something to interpolate between.

Qparams are extracted directly from the packed buffers BEFORE
ModelCompressor.decompress_model runs, because decompression drops
module.weight_packed. Codes and zero-points are shifted from the
signed int4 layout that compressed-tensors uses into the unsigned
uint8 layout that src/quant/qparams.py defines; the dequant formula
scale * (qweight - zeros) is invariant under that shift.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from quant.qparams import QPARAMS_SCHEMA_VERSION, qparams_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_EXCLUDES = ("lm_head", "embed_tokens", "embed")


def _extract_qparams_from_ct_module(module: nn.Module) -> Optional[dict]:
    """Pull qparams out of a compressed-tensors module's packed buffers.

    compressed-tensors stores codes and zeros as signed int4 packed into
    int32 tensors. Our schema (src/quant/qparams.py) uses unsigned uint8
    codes in [0, 2**bits - 1], so we shift codes and zeros by 2**(bits-1)
    before saving. The dequant formula scale * (qweight - zeros) is
    invariant under this shift.
    """
    needed = ("weight_packed", "weight_scale", "weight_zero_point",
              "weight_shape", "quantization_scheme")
    if not all(hasattr(module, a) for a in needed):
        return None

    from compressed_tensors.compressors import unpack_from_int32

    qs = module.quantization_scheme.weights
    bits = int(qs.num_bits)
    if qs.type != "int":
        return None

    packed = module.weight_packed.detach()
    scale = module.weight_scale.detach()
    zp = module.weight_zero_point.detach()
    shape = tuple(int(x) for x in module.weight_shape.detach().tolist())
    n_groups = scale.shape[1] if scale.dim() > 1 else 1

    codes = unpack_from_int32(packed, num_bits=bits,
                              shape=torch.Size(shape), packed_dim=1)
    zeros = unpack_from_int32(zp, num_bits=bits,
                              shape=torch.Size((shape[0], n_groups)),
                              packed_dim=0)

    offset = 1 << (bits - 1)
    qweight = (codes.to(torch.int16) + offset).to(torch.uint8)
    zeros_u = (zeros.to(torch.int16) + offset).to(scale.dtype)

    dtype_str = "bfloat16"
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        dtype_str = str(module.weight.dtype).replace("torch.", "")

    return {
        "qweight":    qweight.contiguous(),
        "scales":     scale.detach().contiguous(),
        "zeros":      zeros_u.contiguous(),
        "perm":       None,
        "bits":       bits,
        "group_size": int(qs.group_size) if qs.group_size else shape[1],
        "sym":        bool(qs.symmetric),
        "perchannel": qs.strategy in ("group", "channel"),
        "act_order":  bool(getattr(qs, "actorder", None)),
        "shape":      shape,
        "dtype":      dtype_str,
        "schema":     QPARAMS_SCHEMA_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a compressed-tensors HF model into the per-layer "
                    "database (one bitwidth slot).",
    )
    parser.add_argument("--model", required=True,
                        help="HF model id or local path of the quantized checkpoint.")
    parser.add_argument("--layer-dir", required=True,
                        help="Database root; the same path you would pass to "
                             "run_quantize.py --save_dir / the search scripts' "
                             "--layer-dir.")
    parser.add_argument("--bitwidth", type=int, required=True,
                        help="Bitwidth slot this checkpoint represents (e.g. 4 for "
                             "a W4 model). Each layer is written to <layer>/<bw>.pth.")
    parser.add_argument("--exclude-patterns", nargs="+", default=list(DEFAULT_EXCLUDES),
                        help="Substring matches on layer names to skip. "
                             f"Default: {' '.join(DEFAULT_EXCLUDES)}.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Dequantized tensor dtype on disk. Default: bfloat16.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing <layer>/<bw>.pth files.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    dtype = getattr(torch, args.dtype)
    layer_dir = Path(args.layer_dir).resolve()
    layer_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.model} on CPU...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        msg = str(e)
        if "compressed-tensors" in msg.lower() or "compressed_tensors" in msg.lower():
            raise SystemExit(
                "Loading failed. The compressed-tensors package is required to "
                "auto-decompress this checkpoint. Install with:\n"
                "    pip install compressed-tensors\n"
                f"Original error: {e}"
            )
        raise

    # Pass 1: snapshot per-layer qparams BEFORE decompression. decompress_model
    # deletes module.weight_packed, so we have to read the integer codes first.
    qparams_by_layer: dict = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in name for p in args.exclude_patterns):
            continue
        qp = _extract_qparams_from_ct_module(module)
        if qp is not None:
            qparams_by_layer[name] = qp

    # Pass 2: decompress so module.weight materializes (compressed-tensors
    # only adds .weight at decompress time).
    try:
        from compressed_tensors import ModelCompressor
        mc = ModelCompressor.from_pretrained_model(model)
        if mc is not None:
            logger.info("Decompressing compressed-tensors weights to dense...")
            mc.decompress_model(model)
    except ImportError:
        logger.warning("compressed_tensors not importable; assuming the model "
                       "is already dense.")
    except Exception as e:
        logger.warning(f"decompress_model failed ({type(e).__name__}: {e}); "
                       f"assuming the model is already dense.")

    # Pass 3: write per-layer .pth (dense bf16) + _qparams.pt (sidecar).
    n_written = n_skipped = n_qparams = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in name for p in args.exclude_patterns):
            n_skipped += 1
            continue

        out = layer_dir / name / f"{args.bitwidth}.pth"
        qp_out = Path(qparams_path(layer_dir / name, args.bitwidth))
        if out.exists() and not args.overwrite:
            logger.warning(f"skip existing {out} (use --overwrite to replace)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        weight = module.weight.detach().cpu().to(dtype)
        torch.save(weight, out)
        n_written += 1

        qp = qparams_by_layer.get(name)
        if qp is not None:
            qp["dtype"] = str(dtype).replace("torch.", "")
            torch.save(qp, qp_out)
            n_qparams += 1

        if n_written % 50 == 0:
            logger.info(f"  wrote {n_written} layers")

    logger.info(f"Done. Wrote {n_written} layer files "
                f"({n_qparams} with qparams sidecars), skipped {n_skipped} "
                f"(matched exclude patterns) under {layer_dir}.")
    if n_written == 0:
        logger.warning("No layers were written. Check --exclude-patterns and the "
                       "model's linear-layer naming.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
