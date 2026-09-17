#!/usr/bin/env bash
# Prepare a worker node from the Mini over SSH. Nothing is copied except the
# ~40 MB training JSONL; code comes from GitHub and the base model from Hugging Face.
# Usage: scripts/cluster/sync_node.sh gauttam@10.20.30.2
set -euo pipefail
cd "$(dirname "$0")/../.."
NODE=${1:?usage: sync_node.sh user@host}
REPO_DIR="$PWD"                       # workers must use the same absolute path
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "== code: same commit, same path ($REPO_DIR) on $NODE"
git push -q origin "$BRANCH"
ssh "$NODE" "set -e
  if [ ! -d '$REPO_DIR/.git' ]; then mkdir -p \"\$(dirname '$REPO_DIR')\" && git clone -q https://github.com/Gsura-n/dictum '$REPO_DIR'; fi
  cd '$REPO_DIR' && git fetch -q && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH
  [ -d .venv ] || /opt/homebrew/bin/python3.11 -m venv .venv
  .venv/bin/pip install -q -e '.[mlx]'
  .venv/bin/python -c 'import mlx.core, mlx_lm; print(\"mlx ok\")'"

echo "== training data (the only thing copied)"
ssh "$NODE" "mkdir -p '$REPO_DIR/evals/data/finetune'"
rsync -az --delete evals/data/finetune/dictation "$NODE:$REPO_DIR/evals/data/finetune/"

echo "== base model (downloaded on the node from Hugging Face, cached)"
ssh "$NODE" "cd '$REPO_DIR' && .venv/bin/python -c \"from huggingface_hub import snapshot_download as s; s('mlx-community/Llama-3.2-3B-Instruct-4bit')\" >/dev/null && echo 'model cached'"
echo "node $NODE ready"
