#!/usr/bin/env bash
#
# RCO MoE expert-pruning search.
# Writes a per-layer prune mask to <save_mask>.pt that
# scripts/run_build_checkpoint.sh can materialize into an HF checkpoint.
#
# Usage:
#   scripts/run_search_prune.sh <hf_model_id> <sparsity> <save_mask> [n_steps]
#
# Example:
#   scripts/run_search_prune.sh allenai/OLMoE-1B-7B-0125-Instruct 0.25 mask.pt 300
#
set -euo pipefail

MODEL="${1:?usage: $0 <model> <sparsity> <save_mask> [n_steps]}"
SPARSITY="${2:?usage: $0 <model> <sparsity> <save_mask> [n_steps]}"
SAVE_MASK="${3:?usage: $0 <model> <sparsity> <save_mask> [n_steps]}"
N_STEPS="${4:-300}"

python rco_search_prune.py \
    --model            "${MODEL}" \
    --target-sparsity  "${SPARSITY}" \
    --n-steps          "${N_STEPS}" \
    --n-gumbel-samples 4 \
    --antithetic \
    --batch-size       4 \
    --calibration-data fineweb_edu \
    --calibration-samples 64 \
    --save-mask        "${SAVE_MASK}"
