#!/usr/bin/env bash
#
# Materialize a HuggingFace checkpoint from an RCO search result.
#
# Usage:
#   quant mode:
#     scripts/run_build_checkpoint.sh quant <model> <assignment.json> <layer_dir> <output>
#   prune mode:
#     scripts/run_build_checkpoint.sh prune <model> <mask.pt> <output>
#
set -euo pipefail

MODE="${1:?usage: $0 quant|prune ...}"
shift

case "${MODE}" in
  quant)
    MODEL="${1:?missing model}"; CONFIG="${2:?missing assignment.json}"
    LAYER_DIR="${3:?missing layer_dir}"; OUTPUT="${4:?missing output dir}"
    python run_build_checkpoint.py quant \
        --model     "${MODEL}" \
        --config    "${CONFIG}" \
        --layer-dir "${LAYER_DIR}" \
        --output    "${OUTPUT}"
    ;;
  prune)
    MODEL="${1:?missing model}"; MASK="${2:?missing mask.pt}"; OUTPUT="${3:?missing output dir}"
    python run_build_checkpoint.py prune \
        --model      "${MODEL}" \
        --prune-mask "${MASK}" \
        --output     "${OUTPUT}"
    ;;
  *)
    echo "unknown mode '${MODE}' (expected: quant | prune)" >&2
    exit 2
    ;;
esac
