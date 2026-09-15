#!/usr/bin/env bash
# LoRA fine-tune of Llama 3.2 3B Instruct (4-bit) for dictation cleanup, on Apple Silicon with MLX.
# Usage: scripts/finetune/train.sh [iters]      e.g. scripts/finetune/train.sh 100   (smoke run)
set -euo pipefail
cd "$(dirname "$0")/../.."
ITERS=${1:-2000}
BASE=${BASE:-mlx-community/Llama-3.2-3B-Instruct-4bit}
ADAPTER=${ADAPTER:-adapters/dictation-llama3.2-3b}

pip install -q 'mlx-lm>=0.31'
python scripts/finetune/prepare_data.py

# Notes on settings:
#  batch 4, lr 1e-4, rank 8 on 16 layers: standard small-data LoRA starting point
#  --mask-prompt: loss only on the cleaned output, not on the prompt we already know
#  --grad-checkpoint + short max-seq-length: fits in 16 GB alongside other apps
python -m mlx_lm lora \
  --model "$BASE" \
  --train \
  --data evals/data/finetune/dictation \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --mask-prompt \
  --max-seq-length 512 \
  --grad-checkpoint \
  --steps-per-report 20 \
  --steps-per-eval 200 \
  --save-every 200 \
  --seed 20260915

echo
echo "adapter saved to $ADAPTER"
echo "evaluate on dev:  dictum eval --suite disflqa --split dev --backend mlx --as-mode dictation_ft"
