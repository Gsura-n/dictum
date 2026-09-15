#!/usr/bin/env bash
# One-shot setup on macOS (Apple Silicon). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup targets Apple Silicon macOS (MLX). Other platforms: see README."; exit 1
fi

PY=${PYTHON:-python3}
echo "using $($PY --version) at $(which $PY)"
$PY -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'

if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python "$PY" 2>/dev/null || true
  source .venv/bin/activate
  uv pip install -e '.[parakeet,whisper,refine,mac,dev]'
else
  [[ -d .venv ]] || $PY -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e '.[parakeet,whisper,refine,mac,dev]'
fi

python scripts/fetch_datasets.py || echo "(eval datasets not fetched; run scripts/fetch_datasets.py later)"

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  dictum devices        # confirm your mic is visible"
echo "  dictum check          # permissions, mic, Ollama models"
echo "  dictum run            # hold Right Option anywhere and talk"
