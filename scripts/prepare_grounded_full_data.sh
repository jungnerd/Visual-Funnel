#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n vf-qwen25 python scripts/prepare_textvqa_data_json.py \
  --input data/textvqa/TextVQA_0.5.1_val.json \
  --output data/textvqa/data_full.json \
  --limit 0

conda run -n vf-qwen25 python scripts/prepare_gqa_subset.py \
  --output-dir data/gqa_full \
  --limit 0 \
  --shuffle \
  --seed 0 \
  --qa-per-image 1

conda run -n vf-qwen25 python scripts/prepare_vqa_subset.py \
  --task docvqa \
  --output-dir data/docvqa_full \
  --limit 0 \
  --shuffle \
  --seed 0

conda run -n vf-qwen25 python scripts/prepare_vqa_subset.py \
  --task infovqa \
  --output-dir data/infovqa_full \
  --limit 0 \
  --shuffle \
  --seed 0
