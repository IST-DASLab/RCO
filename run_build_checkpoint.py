#!/usr/bin/env python3
"""Materialize HF checkpoints from RCO search outputs.

Subcommands:
  quant   Write a quantized HF checkpoint from a bitwidth assignment, either
          as fake-quant bf16 (default) or as a real compressed-tensors packed
          checkpoint (--format compressed-tensors).
  prune   Realize an expert-pruning mask by either zeroing pruned experts and
          their router rows (--prune-mode zero, default) or physically
          removing them and shrinking the gate (--prune-mode remove).
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import resolve_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_EXPERT_RE = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)$")

_PROJ_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# ----------------------------------------------------------------------------
# Quant helpers
# ----------------------------------------------------------------------------


def _parse_config_txt(path: str) -> dict:
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, bw = line.rsplit(":", 1)
            cfg[name.strip()] = int(bw.strip())
    return cfg


def _parse_config_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: int(v) for k, v in data["assignment"].items()}


def _load_weight(layer_dir: Path, layer_name: str, bitwidth: int) -> torch.Tensor:
    pth = layer_dir / layer_name / f"{bitwidth}.pth"
    if not pth.exists():
        raise FileNotFoundError(pth)
    return torch.load(str(pth), map_location="cpu", weights_only=True)


def _fused_expert_target(model: nn.Module, layer_name: str
                         ) -> Optional[Tuple[nn.Module, int, str]]:
    m = _EXPERT_RE.match(layer_name)
    if not m:
        return None
    experts_path, idx_str, sub = m.group(1), m.group(2), m.group(3)
    experts_mod = resolve_module(model, experts_path)
    if isinstance(experts_mod, nn.ModuleList):
        return None
    return experts_mod, int(idx_str), sub


def _apply_fused(experts_mod: nn.Module, expert_idx: int, sub: str,
                 weight: torch.Tensor) -> None:
    if sub == "down_proj":
        target = experts_mod.down_proj.data[expert_idx]
        if target.shape != weight.shape:
            raise ValueError(
                f"down_proj slice {tuple(target.shape)} != "
                f"weight {tuple(weight.shape)}")
        target.copy_(weight.to(experts_mod.down_proj.dtype))
        return

    # gate_up_proj packs [gate; up] along dim 0
    gate_up = experts_mod.gate_up_proj.data
    full = gate_up[expert_idx]
    inter = full.shape[0] // 2
    expected = (inter, full.shape[1])
    if weight.shape != expected:
        raise ValueError(
            f"{sub} slice {expected} != weight {tuple(weight.shape)}")
    sl = slice(0, inter) if sub == "gate_proj" else slice(inter, None)
    full[sl, :].copy_(weight.to(gate_up.dtype))


def _build_compressed_tensors_config(
    assignment: Dict[str, int],
    group_size: int = 128,
    symmetric: bool = False,
    ignore: Optional[list] = None,
    fmt: str = "dense",
    status: str = "frozen",
    actorder: Optional[str] = None,
) -> dict:
    """Build a compressed-tensors quantization_config for a mixed-bitwidth checkpoint.

    The most-frequent bitwidth becomes the default (group_default with
    targets: ["Linear"]); every other bitwidth gets its own config_group
    with a regex listing the affected layer names. fmt and status
    are "dense"/"frozen" for the fake-quant (bf16) output mode and
    "pack-quantized"/"compressed" for the packed-int output mode.
    """
    bw_counts = Counter(assignment.values())
    default_bw = bw_counts.most_common(1)[0][0]

    def weights_spec(bw: int) -> dict:
        return {
            "num_bits": int(bw),
            "group_size": int(group_size),
            "symmetric": bool(symmetric),
            "strategy": "group",
            "type": "int",
            "actorder": actorder,
            "observer": "minmax",
            "dynamic": False,
            "block_structure": None,
        }

    # PackedQuantizationCompressor.can_compress restricts the inferred format to
    # num_bits in (4, 8); without an explicit per-scheme format the 3/5/6 bit
    # layers would fall through to naive_quantized on load, which stores
    # weight (int8) instead of weight_packed and breaks the loader.
    # Forcing pack-quantized per scheme keeps every bitwidth on the same
    # packed-int32 schema.
    per_scheme_format = "pack-quantized" if fmt == "pack-quantized" else None

    def scheme_block(targets: list, bw: int) -> dict:
        block = {"targets": targets, "weights": weights_spec(bw)}
        if per_scheme_format is not None:
            block["format"] = per_scheme_format
        return block

    config_groups: Dict[str, dict] = {
        "group_default": scheme_block(["Linear"], default_bw),
    }

    bw_to_layers: Dict[int, list] = {}
    for layer, bw in assignment.items():
        if bw == default_bw:
            continue
        bw_to_layers.setdefault(bw, []).append(layer)

    # compressed-tensors prefers a list of literal module names over one big
    # alternation regex: the regex form fires a "Could not match ... in
    # instance of <ModelClass>" warning during apply_quantization_config and
    # downstream loaders sometimes refuse the resulting checkpoint.
    for bw in sorted(bw_to_layers):
        layers = sorted(bw_to_layers[bw])
        config_groups[f"group_bw{bw}"] = scheme_block(layers, bw)

    return {
        "quant_method": "compressed-tensors",
        "quantization_status": status,
        "format": fmt,
        "ignore": list(ignore or ["lm_head"]),
        "kv_cache_scheme": None,
        "config_groups": config_groups,
    }


def _inject_quant_config(output_dir: Path, quant_config: dict) -> None:
    cfg_path = output_dir / "config.json"
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    model_cfg["quantization_config"] = quant_config
    with open(cfg_path, "w") as f:
        json.dump(model_cfg, f, indent=2)
    logger.info(f"Injected quantization_config into {cfg_path}")


# ----------------------------------------------------------------------------
# Quant entry
# ----------------------------------------------------------------------------


def _load_base_and_assignment(args):
    """Shared by both quant output formats: load base model + tokenizer,
    derive the per-layer bitwidth assignment from args."""
    logger.info(f"Loading {args.model} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    layer_dir = Path(args.layer_dir)
    if args.uniform_bw is not None:
        assignment = _make_uniform_assignment(model, args.uniform_bw)
    elif args.config.endswith(".json"):
        assignment = _parse_config_json(args.config)
    else:
        assignment = _parse_config_txt(args.config)
    return model, tokenizer, layer_dir, assignment


def _make_uniform_assignment(model, bitwidth: int) -> dict:
    """Enumerate quantizable Linear layers and assign every one the given
    bitwidth. Recognizes both dense layers (have .weight) and pre-quantized
    compressed-tensors layers (have weight_packed instead).
    """
    assignment = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if not any(p in name for p in _PROJ_NAMES):
            continue
        # accept both dense and packed-base modules
        if hasattr(m, "weight") or hasattr(m, "weight_packed"):
            assignment[name] = bitwidth
    return assignment


def run_quant(args: argparse.Namespace) -> int:
    if args.config is None and args.uniform_bw is None:
        raise SystemExit("Provide --config or --uniform-bw.")
    if args.format == "fake-quant":
        return _run_quant_fake_quant(args)
    if args.format == "compressed-tensors":
        return _run_quant_packed(args)
    raise SystemExit(f"unknown --format {args.format!r}")


def _run_quant_fake_quant(args: argparse.Namespace) -> int:
    model, tokenizer, layer_dir, assignment = _load_base_and_assignment(args)

    # if the base model is itself a compressed-tensors checkpoint, transformers
    # keeps it packed (no .weight attribute); decompress in place so the layer
    # loop can overwrite .weight with our chosen-bitwidth tensors.
    try:
        from compressed_tensors import ModelCompressor
        mc = ModelCompressor.from_pretrained_model(model)
        if mc is not None:
            logger.info("Decompressing base model's compressed-tensors weights...")
            mc.decompress_model(model)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"decompress_model failed ({type(e).__name__}: {e}); "
                       f"assuming the base model is already dense.")

    logger.info(f"{len(assignment)} layers to write")
    n_replaced = n_fused = 0
    for name, bw in assignment.items():
        w = _load_weight(layer_dir, name, bw)
        fused = _fused_expert_target(model, name)
        if fused is not None:
            experts_mod, idx, sub = fused
            _apply_fused(experts_mod, idx, sub, w)
            n_fused += 1
        else:
            target = resolve_module(model, name)
            target.weight.data.copy_(w.to(target.weight.dtype))
        n_replaced += 1
        if n_replaced % 200 == 0:
            logger.info(f"  {n_replaced}/{len(assignment)}")

    logger.info(f"Wrote {n_replaced} layers ({n_fused} via fused-experts slicing).")

    # decompress_model leaves weight_scale / weight_zero_point / weight_shape
    # parameters registered on each linear; strip them for a clean bf16 output.
    _strip_leftover_compressed_params(model)
    if getattr(model.config, "quantization_config", None) is not None:
        try:
            del model.config.quantization_config
        except AttributeError:
            pass

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    if not args.no_quant_config:
        quant_config = _build_compressed_tensors_config(
            assignment,
            group_size=args.group_size,
            symmetric=args.symmetric,
            fmt="dense",
            status="frozen",
        )
        _inject_quant_config(output_dir, quant_config)
    return 0


def _run_quant_packed(args: argparse.Namespace) -> int:
    """Pack the chosen bitwidth per layer into a compressed-tensors checkpoint.

    Requires <layer>/<bw>_qparams.pt sidecars in the database (produced by
    run_quantize.py or run_split_checkpoint.py).

    Flow (mirrors the official compressed-tensors save lifecycle):
      1. Load the base model and decompress it to dense if it shipped packed.
      2. Build a QuantizationConfig with per-layer bitwidth groups and call
         apply_quantization_config to attach a quantization_scheme to
         every targeted Linear and initialize empty weight_scale /
         weight_zero_point parameters with the right shape per bitwidth.
      3. For each layer with a qparams sidecar, build the packed state
         (weight_packed, weight_scale, weight_zero_point,
         weight_shape) and install it via replace_direct_state_dict;
         this swaps the dense .weight for the packed layout and removes
         any extras the loader would later flag as unexpected.
      4. Mark each module COMPRESSED and save_pretrained.
      5. Inject the matching quantization_config block into config.json.
    """
    from quant.qparams import load_qparams
    from compressed_tensors import ModelCompressor
    from compressed_tensors.quantization import (
        QuantizationConfig, QuantizationStatus, apply_quantization_config,
    )
    from compressed_tensors.utils import replace_direct_state_dict

    model, tokenizer, layer_dir, assignment = _load_base_and_assignment(args)

    # apply_quantization_config expects clean nn.Linear modules with .weight;
    # if the base was a compressed-tensors checkpoint, decompress it first.
    try:
        mc_base = ModelCompressor.from_pretrained_model(model)
        if mc_base is not None:
            logger.info("Decompressing base model's compressed-tensors weights...")
            mc_base.decompress_model(model)
            _strip_leftover_compressed_params(model)
    except Exception as e:
        logger.warning(f"decompress_model skipped ({type(e).__name__}: {e}); "
                       f"assuming the base model is already dense.")

    # remove any stale quantization_config from the base; we write our own.
    if getattr(model.config, "quantization_config", None) is not None:
        try:
            del model.config.quantization_config
        except AttributeError:
            pass

    # build the config in FROZEN state so apply_quantization_config initializes
    # weight_scale / weight_zero_point parameters with the correct per-bitwidth
    # shape (force_zero_point fires when status < COMPRESSED, which we need).
    apply_cfg_dict = _build_compressed_tensors_config(
        assignment,
        group_size=args.group_size,
        symmetric=args.symmetric,
        fmt="pack-quantized",
        status="frozen",
    )
    qcfg = QuantizationConfig.model_validate(apply_cfg_dict)
    logger.info(f"Applying quantization config to {len(assignment)} layers...")
    apply_quantization_config(model, qcfg)

    logger.info(f"{len(assignment)} layers to repack")
    n_packed = 0
    missing_qparams: list = []
    for name, bw in assignment.items():
        layer_path = layer_dir / name
        try:
            qp = load_qparams(layer_path, bw)
        except FileNotFoundError:
            missing_qparams.append((name, bw))
            continue

        weight_packed, weight_scale, weight_zp, weight_shape = _pack_qparams_for_module(qp)
        mod = resolve_module(model, name)

        new_state: Dict[str, torch.Tensor] = {}
        if getattr(mod, "bias", None) is not None:
            new_state["bias"] = mod.bias.data
        new_state["weight_packed"] = weight_packed
        new_state["weight_scale"] = weight_scale.to(torch.bfloat16)
        if weight_zp is not None:
            new_state["weight_zero_point"] = weight_zp
        new_state["weight_shape"] = weight_shape

        replace_direct_state_dict(mod, new_state)
        mod.quantization_status = QuantizationStatus.COMPRESSED

        n_packed += 1
        if n_packed % 200 == 0:
            logger.info(f"  {n_packed}/{len(assignment)}")

    if missing_qparams:
        sample = ", ".join(f"{n}@{b}" for n, b in missing_qparams[:3])
        more = f" (+{len(missing_qparams) - 3} more)" if len(missing_qparams) > 3 else ""
        raise SystemExit(
            f"compressed-tensors mode needs <bw>_qparams.pt for every layer "
            f"in the assignment, but {len(missing_qparams)} are missing. "
            f"Examples: {sample}{more}. Re-run run_quantize.py or run_split_checkpoint.py "
            f"to populate these sidecars, or use --format fake-quant."
        )

    logger.info(f"Repacked {n_packed} layers from qparams sidecars.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    quant_config = _build_compressed_tensors_config(
        assignment,
        group_size=args.group_size,
        symmetric=args.symmetric,
        fmt="pack-quantized",
        status="compressed",
    )
    _inject_quant_config(output_dir, quant_config)
    return 0


def _pack_qparams_for_module(qp: dict):
    """Convert our qparams bundle into compressed-tensors packed tensors.

    Returns (weight_packed int32, weight_scale, weight_zero_point int32 | None,
    weight_shape int64). The packed_dim/shift conventions mirror what we
    observed on a freshly-loaded compressed-tensors module so the output is
    schema-compatible with compressed_tensors.PackedQuantizationCompressor.
    """
    from compressed_tensors.compressors import pack_to_int32

    bits = qp["bits"]
    offset = 1 << (bits - 1)

    # qweight: uint8 in [0, 2**bits-1]  ->  signed int8 in [-2^(b-1), 2^(b-1)-1]
    codes = (qp["qweight"].to(torch.int16) - offset).to(torch.int8)
    weight_packed = pack_to_int32(codes, num_bits=bits, packed_dim=1).to(torch.int32)
    weight_scale = qp["scales"].clone()

    if qp["sym"]:
        weight_zero_point = None
    else:
        zeros = (qp["zeros"].to(torch.float32).round().to(torch.int16) - offset).to(torch.int8)
        weight_zero_point = pack_to_int32(zeros, num_bits=bits, packed_dim=0).to(torch.int32)

    weight_shape = torch.tensor(qp["shape"], dtype=torch.int64)
    return weight_packed, weight_scale, weight_zero_point, weight_shape


_LEFTOVER_CT_PARAMS = ("weight_packed", "weight_scale", "weight_zero_point",
                       "weight_shape", "weight_g_idx")


def _strip_leftover_compressed_params(model: nn.Module) -> int:
    """Remove the packed-format buffers that decompress_model leaves behind.

    decompress_model materializes a dense .weight on each linear and
    deletes weight_packed only; weight_scale, weight_zero_point
    and weight_shape stay registered as parameters of the module. Without
    this strip, save_pretrained would write a checkpoint that contains
    both a dense .weight and the metadata buffers, which most loaders
    treat as malformed.
    """
    stripped = 0
    for _, mod in model.named_modules():
        for name in _LEFTOVER_CT_PARAMS:
            if name in mod._parameters and mod._parameters[name] is not None:
                del mod._parameters[name]
                stripped += 1
            elif name in mod._buffers and mod._buffers[name] is not None:
                del mod._buffers[name]
                stripped += 1
    if stripped:
        logger.info(f"Stripped {stripped} leftover compressed-tensors buffers.")
    return stripped


# ----------------------------------------------------------------------------
# Prune helpers
# ----------------------------------------------------------------------------


def _zero_experts(experts: nn.Module, pruned_idx: torch.Tensor, n_experts: int) -> int:
    """Zero pruned experts in either ModuleList or fused-3D layouts."""
    if isinstance(experts, nn.ModuleList):
        for j in pruned_idx.tolist():
            for p in experts[j].parameters():
                p.data.zero_()
        return len(pruned_idx)

    zeroed = 0
    for _, p in experts.named_parameters(recurse=False):
        if p.dim() >= 1 and p.shape[0] == n_experts:
            p.data[pruned_idx] = 0
            zeroed += 1
    if zeroed == 0:
        raise RuntimeError(
            f"No per-expert parameters in {type(experts).__name__}. Expected "
            f"either an nn.ModuleList or params with first dim = {n_experts}.")
    return len(pruned_idx)


def _zero_router(gate: nn.Module, pruned_idx: torch.Tensor) -> None:
    if not (hasattr(gate, "weight")
            and isinstance(gate.weight, (nn.Parameter, torch.Tensor))):
        raise RuntimeError(
            f"Router {type(gate).__name__} has no weight parameter to mask.")
    gate.weight.data[pruned_idx] = 0
    if getattr(gate, "bias", None) is not None:
        gate.bias.data[pruned_idx] = 0


def _shrink_param_dim0(module: nn.Module, name: str, kept_idx: torch.Tensor) -> None:
    """Replace module._parameters[name] with a slice along its first dim.

    Used for both the gate.weight / gate.bias (expert rows) and the per-expert
    parameters in a fused-3D experts container. The slice is cloned so the new
    Parameter owns its storage and the old large tensor is free to garbage-collect.
    """
    p = module._parameters[name]
    new_data = p.data[kept_idx].contiguous().clone()
    del module._parameters[name]
    module.register_parameter(
        name, nn.Parameter(new_data, requires_grad=p.requires_grad))


def _remove_experts(block: nn.Module, kept_idx: torch.Tensor,
                    n_experts: int) -> int:
    """Physically drop pruned experts from a block.mlp and shrink block.gate.

    Returns the number of experts kept in this layer.

    Handles:
      - ModuleList-style experts (one nn.Module per expert): pop pruned entries.
      - Fused 3D experts (a single nn.Module whose per-expert parameters all
        share first dim = n_experts): slice each parameter along dim 0.
      - nn.Linear gate (out_features = n_experts): slice weight / bias rows.
      - Class-specific routers that expose a weight parameter of shape
        (n_experts, hidden_dim) but are NOT an nn.Linear (e.g. OlmoeTopKRouter,
        Qwen3MoeGate): slice weight / bias rows in place.
    """
    experts = block.experts
    kept_list = kept_idx.tolist()
    n_kept = len(kept_list)

    # 1. Shrink the gate. We support both nn.Linear and custom routers as long
    # as they expose .weight whose first dim equals n_experts.
    old_gate = block.gate
    if not (hasattr(old_gate, "weight")
            and isinstance(old_gate.weight, (nn.Parameter, torch.Tensor))):
        raise RuntimeError(
            f"Router {type(old_gate).__name__} has no weight parameter to resize.")
    if old_gate.weight.shape[0] != n_experts:
        raise RuntimeError(
            f"Router {type(old_gate).__name__}.weight first dim "
            f"{old_gate.weight.shape[0]} != n_experts {n_experts}; cannot shrink.")

    if isinstance(old_gate, nn.Linear):
        new_gate = nn.Linear(
            old_gate.in_features, n_kept,
            bias=old_gate.bias is not None,
            dtype=old_gate.weight.dtype,
            device=old_gate.weight.device,
        )
        with torch.no_grad():
            new_gate.weight.copy_(old_gate.weight.data[kept_idx])
            if old_gate.bias is not None:
                new_gate.bias.copy_(old_gate.bias.data[kept_idx])
        block.gate = new_gate
    else:
        _shrink_param_dim0(old_gate, "weight", kept_idx)
        if getattr(old_gate, "bias", None) is not None:
            _shrink_param_dim0(old_gate, "bias", kept_idx)

    # 2. Shrink the experts container.
    if isinstance(experts, nn.ModuleList):
        block.experts = nn.ModuleList([experts[i] for i in kept_list])
    else:
        sliced = False
        for name, p in list(experts.named_parameters(recurse=False)):
            if p.dim() >= 1 and p.shape[0] == n_experts:
                _shrink_param_dim0(experts, name, kept_idx)
                sliced = True
        if not sliced:
            raise RuntimeError(
                f"No per-expert parameters in {type(experts).__name__}. "
                f"Expected either an nn.ModuleList or params with first dim = {n_experts}.")

    # 3. Update num_experts-style attributes wherever the implementation puts them.
    for owner in (block, experts, old_gate):
        for attr in ("num_experts", "n_experts", "num_local_experts"):
            if hasattr(owner, attr):
                setattr(owner, attr, n_kept)

    return n_kept


# ----------------------------------------------------------------------------
# Prune entry
# ----------------------------------------------------------------------------


def _resolve_mlp_block(layer: nn.Module) -> Optional[nn.Module]:
    """Find an MoE block on a transformer layer, or None if absent.

    Looks for the conventional 'mlp' attribute and checks that it has both
    'experts' and 'gate' children; otherwise returns None.
    """
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    if not (hasattr(mlp, "experts") and hasattr(mlp, "gate")):
        return None
    return mlp


def run_prune(args: argparse.Namespace) -> int:
    prune_mask = torch.load(args.prune_mask, weights_only=True)
    if prune_mask.dtype != torch.bool:
        prune_mask = prune_mask.bool()

    logger.info(f"Loading {args.model} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_layers = model.config.num_hidden_layers
    n_experts_attr = "num_experts" if hasattr(model.config, "num_experts") else \
                     ("num_local_experts" if hasattr(model.config, "num_local_experts") else None)
    if n_experts_attr is None:
        raise RuntimeError("num_experts / num_local_experts not found in config.")
    n_experts = getattr(model.config, n_experts_attr)
    top_k = getattr(model.config, "num_experts_per_tok", None)

    if prune_mask.shape != (n_layers, n_experts):
        raise ValueError(
            f"prune_mask shape {tuple(prune_mask.shape)} != "
            f"(num_hidden_layers, num_experts) = ({n_layers}, {n_experts})")

    if args.prune_mode == "zero":
        n_pruned = 0
        for l in range(n_layers):
            if not prune_mask[l].any():
                continue
            mlp = _resolve_mlp_block(model.model.layers[l])
            if mlp is None:
                logger.warning(f"Layer {l} has no MoE block; skipping.")
                continue
            pruned_idx = prune_mask[l].nonzero(as_tuple=True)[0]
            _zero_router(mlp.gate, pruned_idx)
            n_pruned += _zero_experts(mlp.experts, pruned_idx, n_experts)
        logger.info(f"Zeroed {n_pruned} expert slots across {n_layers} layers.")
    else:
        # Physical removal: shrink each layer's experts list + gate.
        per_layer_num_experts = []
        total_pruned = 0
        for l in range(n_layers):
            mlp = _resolve_mlp_block(model.model.layers[l])
            if mlp is None:
                per_layer_num_experts.append(n_experts)
                continue
            kept_mask = ~prune_mask[l]
            kept_idx = kept_mask.nonzero(as_tuple=True)[0]
            n_kept = int(kept_idx.numel())
            if top_k is not None and n_kept < top_k:
                raise ValueError(
                    f"Layer {l}: kept {n_kept} < top_k {top_k}; routing would fail.")
            if n_kept == n_experts:
                per_layer_num_experts.append(n_experts)
                continue
            n_kept_actual = _remove_experts(mlp, kept_idx, n_experts)
            per_layer_num_experts.append(n_kept_actual)
            total_pruned += n_experts - n_kept_actual

        uniq = set(per_layer_num_experts)
        if len(uniq) == 1 and uniq.pop() != n_experts:
            new_n = per_layer_num_experts[0]
            setattr(model.config, n_experts_attr, new_n)
            model.config.original_num_experts = n_experts
            logger.info(f"Uniform pruning: rewrote {n_experts_attr} {n_experts} -> {new_n}")
        elif len(uniq) > 1:
            model.config.per_layer_num_experts = per_layer_num_experts
            model.config.original_num_experts = n_experts
            logger.info(
                f"Heterogeneous pruning: wrote per_layer_num_experts "
                f"(min {min(per_layer_num_experts)}, max {max(per_layer_num_experts)}). "
                f"Stock HF / vLLM loaders need a per-layer patch to read this field.")
        logger.info(f"Removed {total_pruned} expert slots across {n_layers} layers.")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sp = p.add_subparsers(dest="mode", required=True)

    pq = sp.add_parser("quant", help="Write a quantized checkpoint.")
    pq.add_argument("--model", required=True)
    pq.add_argument("--config", default=None,
                    help="RCO config (.txt or .json). Omit to use --uniform-bw.")
    pq.add_argument("--uniform-bw", type=int, default=None,
                    help="Apply this bitwidth uniformly to every linear layer.")
    pq.add_argument("--layer-dir", required=True)
    pq.add_argument("--output", required=True)
    pq.add_argument("--format", choices=["fake-quant", "compressed-tensors"],
                    default="fake-quant",
                    help="Output format. 'fake-quant' writes dense bf16 "
                         "weights (already on the integer grid) with a "
                         "quantization_config metadata block; loadable by any "
                         "HF tool. 'compressed-tensors' writes the real "
                         "packed-int format (weight_packed / weight_scale / "
                         "weight_zero_point / weight_shape) using the qparams "
                         "sidecars in the database; on-disk size matches the "
                         "average bitwidth. Default: fake-quant.")
    pq.add_argument("--group-size", type=int, default=128,
                    help="GPTQ group_size recorded in the quantization_config "
                         "block; should match the value used by run_quantize.py "
                         "when the database was built. Default: 128.")
    pq.add_argument("--symmetric", action="store_true",
                    help="Mark the quantization_config as symmetric. Default: "
                         "asymmetric, matching run_quantize.py's default.")
    pq.add_argument("--no-quant-config", action="store_true",
                    help="(fake-quant only) Skip injecting the quantization_config "
                         "metadata block into the output config.json.")
    pq.set_defaults(func=run_quant)

    pp = sp.add_parser("prune", help="Write a pruned checkpoint.")
    pp.add_argument("--model", required=True)
    pp.add_argument("--prune-mask", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--prune-mode", choices=["zero", "remove"], default="zero",
                    help="'zero' (default): keep tensor shapes, zero out pruned "
                         "expert FFN weights and their router rows. Loads in any "
                         "stock HF / vLLM with no patches; on-disk size unchanged. "
                         "'remove': physically drop pruned experts from "
                         "mlp.experts and shrink mlp.gate; rewrites num_experts "
                         "(uniform allocation) or writes per_layer_num_experts "
                         "into config.json (heterogeneous). On-disk size shrinks "
                         "proportional to the kept count; heterogeneous output "
                         "requires a per-layer loader patch for vLLM / HF.")
    pp.set_defaults(func=run_prune)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
