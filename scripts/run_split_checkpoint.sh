#!/usr/bin/env bash
#
# Split a uniformly-quantized HuggingFace checkpoint (compressed-tensors
# format) into the per-layer database as a single bitwidth slot. Use this
# to seed one bitwidth of the database from a pre-existing quantized
# model instead of running run_quantize.py for that bitwidth.
#
# Usage:
#   scripts/run_split_checkpoint.sh <hf_model_id> <layer_dir> <bitwidth>
#
# Example:
#   scripts/run_split_checkpoint.sh anm2211/Llama-3.1-70B-Instruct-2Bit \
#       ~/data/llama_70b/layers 4
#
set -euo pipefail

MODEL="${1:?usage: $0 <hf_model_id> <layer_dir> <bitwidth>}"
LAYER_DIR="${2:?usage: $0 <hf_model_id> <layer_dir> <bitwidth>}"
BITWIDTH="${3:?usage: $0 <hf_model_id> <layer_dir> <bitwidth>}"

python run_split_checkpoint.py \
    --model     "${MODEL}" \
    --layer-dir "${LAYER_DIR}" \
    --bitwidth  "${BITWIDTH}"
