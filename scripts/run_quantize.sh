#!/usr/bin/env bash
#
# Build the multi-bitwidth GPTQ layer database for a dense or MoE model.
# Each linear layer is quantized at every bitwidth in --bitwidth_options and
# saved as <layer>/<bw>.pth (dequantized) plus <layer>/<bw>_qparams.pt (codes
# + scales + zeros + perm). The search drivers consume the .pth files.
#
# Usage:
#   scripts/run_quantize.sh <hf_model_id> <save_dir> [nproc]
#
# Example:
#   scripts/run_quantize.sh Qwen/Qwen3-8B $RCO_DATA_ROOT/qwen3_8b_db 8
#
set -euo pipefail

MODEL="${1:?usage: $0 <hf_model_id> <save_dir> [nproc]}"
SAVE_DIR="${2:?usage: $0 <hf_model_id> <save_dir> [nproc]}"
NPROC="${3:-8}"

torchrun --nproc-per-node="${NPROC}" run_quantize.py \
    --model_name_or_path     "${MODEL}" \
    --quantizable_modules    '.*(q|k|v|o|gate|up|down)_proj$' \
    --pre_block_modules      model.embed_tokens \
    --block_modules          model.layers \
    --post_block_modules     model.norm lm_head \
    --calibration_data       fineweb_edu \
    --calibration_samples    256 \
    --calibration_seq_length 2048 \
    --bitwidth_options       2 3 4 5 6 7 8 \
    --calibration_bitwidth   4 \
    --group_size             128 \
    --save_dir               "${SAVE_DIR}"
