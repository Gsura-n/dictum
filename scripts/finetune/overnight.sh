#!/usr/bin/env bash
# Unattended overnight run: fetch data, build the mixed training set, train the
# full adapter, then evaluate base vs 100-step vs full on every dictation suite.
# Everything is logged to logs/overnight-<timestamp>.log; a summary table is
# written to logs/overnight-summary.md at the end.
#
# Usage (Ollama and Docker closed):  caffeinate -i scripts/finetune/overnight.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p logs
TS=$(date +%Y%m%d-%H%M%S)
LOG=logs/overnight-$TS.log
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

step "install"
pip install -q -e '.[mlx]' || exit 1

step "fetch datasets"
python scripts/fetch_datasets.py || exit 1

if [ ! -f evals/data/disflspeech/train.jsonl ]; then
  echo "WARNING: DisfluencySpeech missing; this run trains on Disfl-QA only" | tee -a logs/overnight-warnings.txt
fi

step "train full adapter on Disfl-QA + DisfluencySpeech"
ROWS=$(python scripts/finetune/prepare_data.py | tee /dev/stderr | awk '/-> train/{gsub(",","",$5); print $5}')
ITERS=$(( ROWS / 4 ))          # one epoch at micro-batch 4
echo "training rows: $ROWS, iters: $ITERS"
SKIP_PREP=1 ADAPTER=adapters/dictation-full scripts/finetune/train.sh "$ITERS" || { echo "TRAINING FAILED"; exit 1; }

step "evaluate"
SUMMARY=logs/overnight-summary.md
echo "# Overnight run $TS" > "$SUMMARY"
for suite in disflqa disflspeech; do
  for adapter in none adapters/dictation-llama3.2-3b adapters/dictation-full; do
    step "eval $suite / dev / adapter=$adapter"
    dictum eval --suite "$suite" --split dev --backend mlx --as-mode dictation_ft \
      --adapter "$adapter" --no-failures --markdown | tee -a "$SUMMARY" || echo "eval failed: $suite $adapter" | tee -a "$SUMMARY"
  done
done
step "eval targeted regression cases (dictation) / full adapter"
dictum eval --suite targeted --mode dictation --backend mlx --as-mode dictation_ft \
  --adapter adapters/dictation-full --markdown | tee -a "$SUMMARY" || true

step "done. summary: $SUMMARY, log: $LOG"
