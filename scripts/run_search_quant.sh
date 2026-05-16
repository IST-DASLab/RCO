#!/usr/bin/env bash
#
# RCO mixed-precision quantization search at a target average bitwidth.
# Reads the per-layer GPTQ database produced by scripts/run_quantize.sh and
# writes an assignment.json with the per-layer bitwidth choice.
#
# Usage:
#   scripts/run_search_quant.sh <hf_model_id> <layer_dir> <target_avg_bits> [n_steps] [n_gumbel_samples]
#
# Example:
#   scripts/run_search_quant.sh Qwen/Qwen3-8B $RCO_DATA_ROOT/qwen3_8b_db 2.5 200 4
#
set -euo pipefail

MODEL="${1:?usage: $0 <model> <layer_dir> <target_bits> [n_steps] [n_gumbel]}"
LAYER_DIR="${2:?usage: $0 <model> <layer_dir> <target_bits> [n_steps] [n_gumbel]}"
TARGET_BITS="${3:?usage: $0 <model> <layer_dir> <target_bits> [n_steps] [n_gumbel]}"
N_STEPS="${4:-200}"
N_GUMBEL="${5:-4}"

python rco_search_quant.py \
    --model              "${MODEL}" \
    --layer-dir          "${LAYER_DIR}" \
    --bitwidths          2,3,4,5,6,7,8 \
    --target-avg-bits    "${TARGET_BITS}" \
    --n-steps            "${N_STEPS}" \
    --lr                 0.1 \
    --tau-min            0.01 \
    --n-gumbel-samples   "${N_GUMBEL}" \
    --batch-size         1 \
    --batches-per-step   1 \
    --gradient-checkpointing \
    --weightstore-cache  lazy \
    --calibration-data   fineweb_edu \
    --calibration-samples 256 \
    --device-map         balanced \
    --save-json
