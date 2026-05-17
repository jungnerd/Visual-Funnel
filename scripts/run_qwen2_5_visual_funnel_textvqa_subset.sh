#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MAX_SAMPLES="${MAX_SAMPLES:-20}"
SAVE_PATH="${SAVE_PATH:-./playground/data/results_subset}"

conda run -n vf-qwen25 python run.py \
  --model qwen2_5 \
  --task textvqa \
  --max_samples "${MAX_SAMPLES}" \
  --save_path "${SAVE_PATH}" \
  --overwrite \
  "$@"
