#!/usr/bin/env python3
"""RCO mixed-precision quantization search."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import cleanup_memory, get_layer_param_counts
from data import load_calibration_data
from metrics import compute_baseline_topk, compute_reference_log_probs
from grouping import (
    build_layer_groups,
    build_moe_per_expert_groups,
    parse_group_spec,
)
from models import get_tokenizer, load_model
from search.quant import (
    InterpolatedModel,
    evaluate_assignment,
    get_actual_bitwidth,
    optimize_projected_gumbel,
    parse_bitwidth_map,
    parse_bitwidths,
    round_with_budget_dp,
)
from store import LoadMode, WeightStore
from unfuse import unfuse_moe_experts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Projected Gumbel-softmax bitwidth assignment"
    )

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer-dir', type=str, required=True)
    parser.add_argument('--bitwidths', type=str, default=None,
                        help="Comma-separated bitwidths, e.g. '4,5,6,7,8'")
    parser.add_argument('--high-bw', type=int, default=None)
    parser.add_argument('--low-bw', type=int, default=None)
    parser.add_argument('--target-avg-bits', type=float, required=True)

    parser.add_argument('--objective', type=str, default='kl', choices=['ce', 'kl'])

    parser.add_argument('--n-steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--tau-init', type=float, default=1.0)
    parser.add_argument('--tau-min', type=float, default=0.01)
    parser.add_argument('--n-gumbel-samples', type=int, default=1)
    parser.add_argument('--batches-per-step', type=int, default=1,
                        help="Random batches per step (0=all batches)")
    parser.add_argument('--init', type=str, default='sensitivity',
                        choices=['sensitivity', 'uniform'],
                        help="Alpha init: 'sensitivity' (per-group MSE), "
                             "'uniform' (same logit per option, budget-shifted)")

    parser.add_argument('--calibration-data', type=str, default='fineweb_edu',
                        choices=['c4', 'wikitext2', 'fineweb_edu',
                                 'evol_codealpaca', 'tulu_math'],
                        help="Plain-text datasets use a uniform loss mask; "
                             "evol_codealpaca and tulu_math use an answer-only "
                             "mask derived from the chat template.")
    parser.add_argument('--calibration-samples', type=int, default=256)
    parser.add_argument('--calibration-seq-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)

    parser.add_argument('--layer-groups', type=str, default='')
    parser.add_argument('--moe-per-expert', action='store_true', default=False,
                        help="MoE-aware grouping: one group per (layer, expert) "
                             "with gate/up/down sharing one bw. Non-expert layers "
                             "still use --layer-groups patterns.")

    parser.add_argument('--gradient-checkpointing', action='store_true', default=False)
    parser.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing',
                        action='store_false')

    parser.add_argument('--device-map', type=str, default='balanced',
                        choices=['auto', 'balanced', 'single'],
                        help="'auto' to shard across GPUs, 'single' for GPU 0 only")
    parser.add_argument('--max-memory-gb', type=float, default=None,
                        help="Override per-GPU max memory in GB for model loading. "
                             "Uniform across all GPUs.")
    parser.add_argument('--max-memory-gb-list', type=str, default=None,
                        help="Per-GPU max memory in GB as comma-separated list "
                             "(one entry per visible GPU).")
    parser.add_argument('--cpu-deltas', action='store_true', default=False,
                        help="Keep weight deltas on CPU and stream to GPU each forward "
                             "(halves GPU memory at the cost of PCIe transfers).")
    parser.add_argument('--offload-folder', type=str, default=None,
                        help="Folder for disk offload during model load. "
                             "Required for some MoE models (Qwen3-Next).")

    parser.add_argument('--weightstore-cache', type=str, default='eager',
                        choices=['off', 'lazy', 'eager'])
    parser.add_argument('--bitwidth-map', type=str, default='')
    parser.add_argument('--save-json', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--grad-debug', action='store_true', default=False)
    parser.add_argument('--grad-debug-interval', type=int, default=10)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.bitwidths:
        bitwidths = parse_bitwidths(args.bitwidths)
    elif args.high_bw is not None and args.low_bw is not None:
        bitwidths = sorted({args.low_bw, args.high_bw})
    else:
        raise SystemExit("Specify --bitwidths or both --high-bw and --low-bw")

    if len(bitwidths) < 2:
        raise SystemExit("Need at least 2 distinct bitwidths")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layer_dir = Path(args.layer_dir)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)

    device_map = None if args.device_map == 'single' else args.device_map
    n_gpus = torch.cuda.device_count()
    logger.info(f"CUDA devices: {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    group_patterns = parse_group_spec(args.layer_groups) if args.layer_groups.strip() else []
    group_hash = (hashlib.md5(args.layer_groups.encode()).hexdigest()[:8]
                  if args.layer_groups.strip() else "nogrp")
    bw_tag = "-".join(str(b) for b in bitwidths)
    out_stem = (
        f"projected_gumbel_{args.objective}_bw{bw_tag}"
        f"_target{args.target_avg_bits}"
        f"_{args.calibration_data}_{args.calibration_samples}s"
        f"_grp{group_hash}"
    )
    out_path = layer_dir / f"{out_stem}.txt"
    out_json_path = layer_dir / f"{out_stem}.json"

    logger.info("Loading model...")
    n_bw = len(bitwidths)
    if args.max_memory_gb is not None and args.max_memory_gb_list is not None:
        raise ValueError("--max-memory-gb and --max-memory-gb-list are mutually exclusive")
    if device_map is not None and n_gpus > 1:
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        if args.max_memory_gb_list is not None:
            per_gpu_list = [float(x.strip()) for x in args.max_memory_gb_list.split(',')]
            if len(per_gpu_list) != n_gpus:
                raise ValueError(
                    f"--max-memory-gb-list has {len(per_gpu_list)} entries but "
                    f"{n_gpus} GPUs are visible")
            max_memory = {i: int(g * 1e9) for i, g in enumerate(per_gpu_list)}
        elif args.max_memory_gb is not None:
            per_gpu_bytes = int(args.max_memory_gb * 1e9)
            max_memory = {i: per_gpu_bytes for i in range(n_gpus)}
        else:
            # leave room for n_bw-1 delta copies plus ~2x activations per GPU,
            # but floor at 15% of total to keep the model off disk offload.
            per_gpu_bytes = int(gpu_total * 0.85) // (n_bw + 3)
            per_gpu_bytes = max(per_gpu_bytes, int(gpu_total * 0.15))
            max_memory = {i: per_gpu_bytes for i in range(n_gpus)}
    else:
        max_memory = None

    model = load_model(args.model, device_map=device_map, max_memory=max_memory,
                       offload_folder=args.offload_folder)

    try:
        replaced = unfuse_moe_experts(model)
        if replaced:
            logger.info(f"Unfused {len(replaced)} MoE expert modules")
    except Exception as e:
        logger.warning(f"Could not unfuse experts: {e}")

    tokenizer = get_tokenizer(args.model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    logger.info("Loading calibration data...")
    calibration_data, calibration_masks_full = load_calibration_data(
        args.calibration_data,
        args.calibration_samples,
        args.calibration_seq_length,
        tokenizer, seed=args.seed,
    )
    # For plain text datasets the mask is all-ones, which is equivalent to
    # not applying a mask in compute_kl_loss; pass None so downstream callers
    # skip the (no-op) masked branch.
    if args.calibration_data in {"c4", "wikitext2", "fineweb_edu"}:
        calibration_masks = None
    else:
        calibration_masks = calibration_masks_full

    ws_cache = args.weightstore_cache != 'off'
    ws_mode = LoadMode.EAGER if args.weightstore_cache == 'eager' else LoadMode.LAZY
    weight_store = WeightStore(str(layer_dir), mode=ws_mode, cache=ws_cache).load()

    layer_names = weight_store.get_layer_names()
    param_counts = get_layer_param_counts(model, layer_names)
    if args.moe_per_expert:
        layer_groups = build_moe_per_expert_groups(layer_names, group_patterns)
    else:
        layer_groups = build_layer_groups(layer_names, group_patterns)

    for name in layer_names:
        available = weight_store.get_available_bitwidths(name)
        for bw in bitwidths:
            if bw == 0:
                continue
            if bw not in available:
                logger.error(f"Layer {name} missing bitwidth {bw}. Available: {available}")
                return 1

    logger.info(f"Layers: {len(layer_names)}, Groups: {len(layer_groups)}")
    logger.info(f"Bitwidths: {bitwidths}, Target: {args.target_avg_bits}")

    baseline_path = (
        layer_dir
        / f"baseline_topk{args.top_k}_{args.calibration_data}"
          f"_{args.calibration_samples}x{args.calibration_seq_length}.pt"
    )
    if baseline_path.exists():
        baseline_data = torch.load(str(baseline_path))
        baseline_topk_vals = baseline_data['topk_vals']
        baseline_topk_idx = baseline_data['topk_idx']
    else:
        baseline_topk_vals, baseline_topk_idx = compute_baseline_topk(
            model, calibration_data, args.batch_size, args.top_k,
            masks=calibration_masks
        )
        torch.save(
            {'topk_vals': baseline_topk_vals, 'topk_idx': baseline_topk_idx},
            str(baseline_path),
        )

    ref_log_probs = None
    if args.objective == 'kl':
        model_slug = args.model.replace('/', '_').replace('-', '_')
        ref_lp_path = (
            layer_dir
            / f"ref_log_probs_{model_slug}_{args.calibration_data}"
              f"_{args.calibration_samples}x{args.calibration_seq_length}"
              f"_bs{args.batch_size}_seed{args.seed}.pt"
        )
        if ref_lp_path.exists():
            ref_log_probs = torch.load(str(ref_lp_path))
        else:
            ref_log_probs = compute_reference_log_probs(
                model, calibration_data, args.batch_size
            )

    interp = InterpolatedModel(
        model, weight_store, bitwidths, layer_names, layer_groups,
        param_counts, bitwidth_map=bitwidth_map,
        cpu_deltas=args.cpu_deltas,
    )
    interp.setup()
    if args.init == 'uniform':
        interp.init_alpha_to_bits(args.target_avg_bits)
    else:
        interp.init_alpha_from_sensitivity(args.target_avg_bits)

    logger.info("=" * 60)
    logger.info("PROJECTED GUMBEL-SOFTMAX OPTIMIZATION")
    logger.info("=" * 60)

    t_start = time.time()
    prob_dict, history = optimize_projected_gumbel(
        interp, calibration_data, args.target_avg_bits,
        n_steps=args.n_steps, lr=args.lr,
        tau_init=args.tau_init, tau_min=args.tau_min,
        n_gumbel_samples=args.n_gumbel_samples,
        batch_size=args.batch_size,
        batches_per_step=args.batches_per_step,
        masks=calibration_masks,
        log_interval=args.log_interval,
        objective=args.objective,
        ref_log_probs=ref_log_probs,
        grad_debug=args.grad_debug,
        grad_debug_interval=args.grad_debug_interval,
    )
    t_elapsed = time.time() - t_start
    logger.info(f"Optimization done in {t_elapsed:.1f}s")

    interp.cleanup()
    assignment = round_with_budget_dp(
        prob_dict, param_counts, bitwidths, args.target_avg_bits,
        layer_groups, bitwidth_map=bitwidth_map,
    )

    total_params = sum(param_counts.get(n, 0) for n in layer_names)
    actual_avg_bits = (sum(
        get_actual_bitwidth(assignment[n], bitwidth_map) * param_counts.get(n, 0)
        for n in layer_names
    ) / total_params) if total_params > 0 else 0.0

    bw_counts = Counter(assignment[n] for n in layer_names)
    for bw in sorted(bw_counts):
        logger.info(f"  {bw}-bit: {bw_counts[bw]} layers")
    logger.info(f"Actual avg bits: {actual_avg_bits:.3f} (target: {args.target_avg_bits})")

    del model
    cleanup_memory()
    model = load_model(args.model, device_map=device_map)

    metrics = evaluate_assignment(
        model, weight_store, assignment,
        calibration_data, baseline_topk_vals, baseline_topk_idx,
        args.batch_size, args.top_k, masks=calibration_masks,
    )

    total_bits = sum(
        get_actual_bitwidth(assignment[n], bitwidth_map) * param_counts.get(n, 0)
        for n in layer_names
    )
    bw_layer_counts = Counter(assignment[n] for n in layer_names)
    with open(out_path, 'w') as f:
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Layer directory: {layer_dir}\n")
        f.write(f"# Objective: {args.objective}\n")
        f.write(f"# Bitwidths: {bitwidths}\n")
        if bitwidth_map:
            f.write(f"# Bitwidth map: {bitwidth_map}\n")
        f.write(f"# Steps: {args.n_steps}, LR: {args.lr}\n")
        f.write(f"# Tau: {args.tau_init} -> {args.tau_min}\n")
        f.write(f"# Calibration: {args.calibration_data} "
                f"({args.calibration_samples}x{args.calibration_seq_length})\n")
        f.write(f"# Average bitwidth: {actual_avg_bits:.4f}\n")
        f.write(f"# Total params: {total_params}\n")
        f.write(f"# Total bits: {total_bits:.0f}\n")
        f.write(f"# Final KL: {metrics['kl']:.6f}\n")
        f.write(f"# Final NLL: {metrics['nll']:.4f}\n")
        f.write(f"# Optimization time: {t_elapsed:.1f}s\n")
        f.write(f"#\n")
        for bw in sorted(bw_layer_counts):
            pct = 100.0 * bw_layer_counts[bw] / len(layer_names)
            f.write(f"#   {bw}-bit: {bw_layer_counts[bw]} layers ({pct:.1f}%)\n")
        f.write(f"#\n")
        for name in layer_names:
            f.write(f"{name}: {assignment[name]}\n")
    logger.info(f"Config saved to {out_path}")

    if args.save_json:
        output = {
            'assignment': assignment,
            'prob_dict': {n: {str(k): v for k, v in p.items()}
                          for n, p in prob_dict.items()},
            'actual_avg_bits': actual_avg_bits,
            'target_avg_bits': args.target_avg_bits,
            'bitwidths': bitwidths,
            'objective': args.objective,
            'metrics': metrics,
            'optimization_time_s': t_elapsed,
            'model': args.model,
            'calibration_dataset': args.calibration_data,
            'calibration_samples': args.calibration_samples,
            'n_layers': len(layer_names),
            'n_groups': len(layer_groups),
            'config': {
                'n_steps': args.n_steps, 'lr': args.lr,
                'tau_init': args.tau_init, 'tau_min': args.tau_min,
                'n_gumbel_samples': args.n_gumbel_samples,
            },
            'loss_history': history,
        }
        if bitwidth_map:
            output['bitwidth_map'] = {str(k): v for k, v in bitwidth_map.items()}
        with open(out_json_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON saved to {out_json_path}")

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Bitwidths:   {bitwidths}")
    logger.info(f"Target bits: {args.target_avg_bits}")
    logger.info(f"Actual bits: {actual_avg_bits:.3f}")
    logger.info(f"NLL:         {metrics['nll']:.4f}")
    logger.info(f"KL:          {metrics['kl']:.6f}")
    logger.info(f"Time:        {t_elapsed:.1f}s")
    logger.info(f"Output:      {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
