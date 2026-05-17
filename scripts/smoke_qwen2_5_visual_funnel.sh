#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

conda run -n vf-qwen25 python scripts/smoke_visual_funnel.py \
  --model qwen2_5 \
  "$@"
