#!/usr/bin/env bash
# LoRA fine-tune of Llama 3.2 3B Instruct (4-bit) for dictation cleanup, on Apple Silicon with MLX.
# Usage: scripts/finetune/train.sh [iters]      e.g. scripts/finetune/train.sh 100   (smoke run)
set -euo pipefail
cd "$(dirname "$0")/../.."
ITERS=${1:-2160}         # ~1 epoch at micro-batch 4 over 8,641 rows (overnight.sh computes this from the data)
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}         # effective batch = BATCH * ACCUM = 16
BASE=${BASE:-mlx-community/Llama-3.2-3B-Instruct-4bit}
ADAPTER=${ADAPTER:-adapters/dictation-llama3.2-3b}

pip install -q 'mlx-lm>=0.31'
[ "${SKIP_PREP:-0}" = 1 ] || python scripts/finetune/prepare_data.py

# Notes on settings:
#  lr 1e-4, rank 8 on 16 layers: standard small-data LoRA starting point
#  micro-batch 4 x grad accumulation 4, with grad checkpointing: a straight batch of 16
#    without checkpointing ran out of GPU memory on a 16 GB M4 (peak 11.6 GB, then OOM on
#    a long batch) and was no faster in tokens/sec, so the small micro-batch costs nothing
#  --mask-prompt: loss only on the cleaned output, not on the prompt we already know
python -m mlx_lm lora \
  --model "$BASE" \
  --train \
  --data evals/data/finetune/dictation \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --grad-accumulation-steps "$ACCUM" \
  --grad-checkpoint \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --mask-prompt \
  --max-seq-length 512 \
  --steps-per-report 20 \
  --steps-per-eval 500 \
  --save-every 200 \
  --seed 20260915

echo
echo "adapter saved to $ADAPTER"
echo "evaluate on dev:  dictum eval --suite disflqa --split dev --backend mlx --as-mode dictation_ft"
