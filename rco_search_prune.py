#!/usr/bin/env python3
"""RCO MoE expert-pruning search."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from models import get_tokenizer, load_model
from data import load_calibration_data
from search.prune import (
    build_ref_cache,
    compute_frequency,
    compute_router_scores,
    evaluate_with_mask,
    frequency_uniform_mask,
    get_moe_info,
    optimize,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Projected Gumbel-STE expert pruning for MoE models")

    parser.add_argument('--model', default='allenai/OLMoE-1B-7B-0125-Instruct')
    parser.add_argument('--sparsity', '--target-sparsity',
                        dest='sparsity', type=float, default=0.25,
                        help="Fraction of experts to prune (e.g. 0.25, 0.5)")
    parser.add_argument('--n-steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--tau-init', type=float, default=1.0)
    parser.add_argument('--tau-min', type=float, default=0.05)
    parser.add_argument('--n-gumbel-samples', type=int, default=4)

    parser.add_argument('--calibration-data', default='fineweb_edu')
    parser.add_argument('--calibration-samples', type=int, default=64)
    parser.add_argument('--calibration-seq-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--kl-topk', type=int, default=20,
                        help="KL over top-k reference tokens per position. "
                             "0 = full vocabulary.")
    parser.add_argument('--entropy-reg', type=float, default=0.0,
                        help="Entropy regularization weight.")
    parser.add_argument('--antithetic', action='store_true',
                        help="Antithetic Gumbel sampling for lower variance.")
    parser.add_argument('--router-init', action='store_true',
                        help="Initialize alpha from router gate-weight sums "
                             "(a router-weighted frequency prior).")
    parser.add_argument('--router-scores-file', type=str, default=None,
                        help="Pre-computed router score cache (.pt).")
    parser.add_argument('--router-spread', type=float, default=5.0)
    parser.add_argument('--per-layer-budget', action='store_true',
                        help="Enforce constant per-layer budget.")

    parser.add_argument('--eval-mask', type=str, default=None,
                        help="Eval-only mode: load prune mask from .pt")
    parser.add_argument('--skip-freq-baseline', action='store_true')

    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--max-memory-per-gpu', type=int, default=None,
                        help="Max memory per GPU in GiB. None = auto.")
    parser.add_argument('--offload-folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--save-json', type=str, default=None)
    parser.add_argument('--save-mask', type=str, default=None,
                        help="Save prune mask as .pt file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"Loading model: {args.model}")
    max_memory = None
    if args.max_memory_per_gpu:
        n_gpus = torch.cuda.device_count()
        max_memory = {i: f'{args.max_memory_per_gpu}GiB' for i in range(n_gpus)}
    model = load_model(args.model, device_map=args.device_map,
                       max_memory=max_memory,
                       offload_folder=args.offload_folder)
    tokenizer = get_tokenizer(args.model)

    n_layers, n_experts, top_k = get_moe_info(model)
    N = n_layers * n_experts
    target_budget = int(args.sparsity * N)

    logger.info(f"MoE: {n_layers} layers, {n_experts} experts/layer, top-{top_k}")
    logger.info(f"Sparsity {args.sparsity} -> prune {target_budget}/{N} experts")

    if args.eval_mask:
        prune_mask = torch.load(args.eval_mask)
        cal_data, cal_masks = load_calibration_data(
            args.calibration_data, args.calibration_samples,
            args.calibration_seq_length, tokenizer, args.seed)
        ref_log_probs, ref_masks = build_ref_cache(
            model, cal_data, cal_masks, args.batch_size)
        kl = evaluate_with_mask(
            model, prune_mask, cal_data, ref_log_probs, ref_masks,
            args.batch_size)
        logger.info(f"KL={kl:.6f}")
        return 0

    cal_data, cal_masks = load_calibration_data(
        args.calibration_data, args.calibration_samples,
        args.calibration_seq_length, tokenizer, args.seed)
    ref_log_probs, ref_masks = build_ref_cache(
        model, cal_data, cal_masks, args.batch_size)

    freq_kl = None
    if not args.skip_freq_baseline:
        freq = compute_frequency(
            model, cal_data, n_layers, n_experts, top_k, args.batch_size)
        freq_mask = frequency_uniform_mask(
            freq, n_layers, n_experts, target_budget)
        freq_kl = evaluate_with_mask(
            model, freq_mask, cal_data, ref_log_probs, ref_masks,
            args.batch_size)
        logger.info(f"Freq uniform: KL={freq_kl:.6f}")

    router_scores = None
    if args.router_init:
        if args.router_scores_file and os.path.exists(args.router_scores_file):
            router_scores = torch.load(args.router_scores_file)['router_scores']
        else:
            router_scores = compute_router_scores(
                model, cal_data, n_layers, n_experts, args.batch_size)
            if args.router_scores_file:
                os.makedirs(os.path.dirname(args.router_scores_file) or '.',
                            exist_ok=True)
                torch.save({'router_scores': router_scores},
                           args.router_scores_file)

    logger.info("=" * 60)
    logger.info("PROJECTED GUMBEL-STE EXPERT PRUNING")
    logger.info("=" * 60)

    t0 = time.time()
    alpha, search_mask, history = optimize(
        model, cal_data, ref_log_probs, ref_masks,
        n_layers, n_experts, target_budget,
        n_steps=args.n_steps, lr=args.lr,
        tau_init=args.tau_init, tau_min=args.tau_min,
        n_gumbel_samples=args.n_gumbel_samples,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        kl_topk=args.kl_topk,
        entropy_reg=args.entropy_reg,
        antithetic=args.antithetic,
        router_scores=router_scores,
        router_spread=args.router_spread,
        per_layer_budget=args.per_layer_budget)
    elapsed = time.time() - t0

    n_pruned = search_mask.sum().item()
    per_layer = search_mask.sum(dim=1).tolist()
    search_kl = evaluate_with_mask(
        model, search_mask, cal_data, ref_log_probs, ref_masks, args.batch_size)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model:    {args.model}")
    logger.info(f"Sparsity: {args.sparsity} ({target_budget} experts)")
    logger.info(f"Time:     {elapsed:.1f}s")
    if freq_kl is not None:
        logger.info(f"Freq uniform:  KL={freq_kl:.6f}")
    logger.info(f"STE search:    KL={search_kl:.6f}")
    for l in range(n_layers):
        r = per_layer[l]
        logger.info(f"  Layer {l:2d}: prune {r:2d}/{n_experts} "
                    f"(density {1.0 - r/n_experts:.3f})")

    if args.save_mask:
        torch.save(search_mask.cpu(), args.save_mask)
        logger.info(f"Saved prune mask to {args.save_mask}")
    if args.save_json:
        res = {
            'model': args.model,
            'sparsity': args.sparsity,
            'target_budget': target_budget,
            'calibration_data': args.calibration_data,
            'freq_baseline': {'kl': freq_kl}
                if freq_kl is not None else None,
            'search': {
                'kl': search_kl,
                'per_layer': per_layer,
                'n_pruned': int(n_pruned),
            },
            'time_s': elapsed,
            'config': vars(args),
            'history': history,
        }
        with open(args.save_json, 'w') as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved results to {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
