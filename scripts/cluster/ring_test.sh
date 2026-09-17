#!/usr/bin/env bash
# Verify the cluster before training: both ranks connect and sync a LoRA-sized gradient.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mlx.launch --backend ring --hostfile "${HOSTFILE:-scripts/cluster/hosts.json}" --cwd "$PWD" \
  --python "$PWD/.venv/bin/python" -- scripts/cluster/ring_test.py
