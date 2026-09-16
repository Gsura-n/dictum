#!/usr/bin/env bash
# LoRA fine-tune of Llama 3.2 3B Instruct (4-bit) for dictation cleanup, on Apple Silicon with MLX.
# Usage: scripts/finetune/train.sh [iters]      e.g. scripts/finetune/train.sh 100   (smoke run)
set -euo pipefail
cd "$(dirname "$0")/../.."
ITERS=${1:-540}          # ~1 epoch at batch 16 over 8,641 rows
BATCH=${BATCH:-16}
BASE=${BASE:-mlx-community/Llama-3.2-3B-Instruct-4bit}
ADAPTER=${ADAPTER:-adapters/dictation-llama3.2-3b}

pip install -q 'mlx-lm>=0.31'
python scripts/finetune/prepare_data.py

# Notes on settings:
#  lr 1e-4, rank 8 on 16 layers: standard small-data LoRA starting point
#  batch 16, no grad checkpointing: the 100-step smoke run peaked at 3.2 GB with
#    batch 4 + checkpointing, so memory is not the constraint on a 16 GB M4; speed is
#  --mask-prompt: loss only on the cleaned output, not on the prompt we already know
python -m mlx_lm lora \
  --model "$BASE" \
  --train \
  --data evals/data/finetune/dictation \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --mask-prompt \
  --max-seq-length 512 \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --save-every 100 \
  --seed 20260915

echo
echo "adapter saved to $ADAPTER"
echo "evaluate on dev:  dictum eval --suite disflqa --split dev --backend mlx --as-mode dictation_ft"
