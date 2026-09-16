#!/usr/bin/env python3
"""Publish a trained dictation adapter to the Hugging Face Hub with a model card.

The card's results table is generated from saved eval result files, so every
number on the card traces to a JSON file in evals/results/.

Usage:
  huggingface-cli login
  python scripts/publish_adapter.py --adapter adapters/dictation-v2 \\
      --repo Gsura-n/dictum-dictation-llama3.2-3b-lora \\
      --results evals/results/<disflqa-test>.json evals/results/<disflspeech-test>.json \\
      [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CARD = """---
base_model: {base}
library_name: mlx
license: llama3.2
tags: [mlx, lora, dictation, disfluency-removal, speech, apple-silicon]
language: [en]
---

# Dictum dictation cleanup adapter (Llama 3.2 3B, LoRA)

LoRA adapter that turns raw speech-to-text output into clean written text:
removes fillers and false starts, applies spoken self-corrections ("Tuesday, no
wait, Thursday" becomes "Thursday"), and leaves already-clean text alone. It is
the default refinement model in [Dictum](https://github.com/Gsura-n/dictum), a
local, private dictation tool for macOS.

**Built with Llama.** This adapter is applied on top of `{base}` and is subject
to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) and
[Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy/).

## Results

Word-error-based pass rate on held-out **test** splits (never used for training
or tuning). A case passes when at most 1 word in 10 differs from the human
reference. 95% Wilson confidence intervals. Measured on a Mac Mini M4, 16 GB.

{table}

## Use

```bash
git clone https://github.com/Gsura-n/dictum && cd dictum && ./scripts/setup.sh
dictum run
```

Or directly with mlx-lm:

```python
from mlx_lm import load, generate
model, tok = load("{base}", adapter_path="<download of this repo>")
```

The prompt format matters; see `dictation_ft` in Dictum's `config/default.yaml`.

## Training

- Method: LoRA (rank 8, 16 layers, lr 1e-4, micro-batch 4 x 4 accumulation, prompt masked), one epoch, MLX on Apple Silicon.
- Data ({rows} rows), training splits only:
  - [Disfl-QA](https://github.com/google-research-datasets/Disfl-QA) (Google Research, CC BY 4.0): human-written disfluent questions.
  - [DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech) (AMAAI Lab, Apache 2.0): Switchboard utterances with human cleanup levels.
  - [Switchboard Dialog Act Corpus](https://github.com/cgpotts/swda) (Jurafsky et al. 1997, distribution GPL-2.0): conversational turns with human disfluency markup, converted to pairs by rule.
- Leak controls: exact matches against every dev/test split removed; SwDA turns sharing any 8-word sequence with DisfluencySpeech dev/test removed.
- No AI-generated training text.

## Limitations

- English only. Trained on typed questions and American telephone speech; other accents, domains and very long dictation are less tested.
- Does not handle spoken punctuation, email addresses or URLs by itself; Dictum applies deterministic rules for those before the model runs.
- Can occasionally drop a word that looks like a filler but carries meaning.
- The Switchboard-derived portions originate from LDC-distributed transcripts; review the source licenses above for your use case.
"""


def table(results_files: list[Path]) -> str:
    lines = ["| suite | n | pass | 95% CI | mean WER | p50 latency |", "|---|---|---|---|---|---|"]
    for f in results_files:
        d = json.loads(f.read_text())
        for s in d["summary"]:
            if s.get("split") != "test":
                print(f"WARNING: {f.name} is split '{s.get('split')}', not test", file=sys.stderr)
            lines.append(f"| {s['suite']} | {s['n']} | {s['pass']:.0%} | {s['ci_low']:.0%} to {s['ci_high']:.0%} | "
                         f"{s.get('mean_wer', float('nan')):.3f} | {s['p50_s']:.2f} s |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--base", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adapter = ROOT / args.adapter
    for name in ("adapters.safetensors", "adapter_config.json"):
        if not (adapter / name).exists():
            raise SystemExit(f"missing {adapter / name}")
    report = ROOT / "evals" / "data" / "finetune" / "dictation" / "report.json"
    rows = json.loads(report.read_text())["total_rows"] if report.exists() else "?"

    stage = Path(tempfile.mkdtemp(prefix="dictum-publish-"))
    for name in ("adapters.safetensors", "adapter_config.json"):
        shutil.copy(adapter / name, stage / name)
    (stage / "README.md").write_text(CARD.format(base=args.base, table=table(args.results), rows=rows))
    if report.exists():
        shutil.copy(report, stage / "training_data_report.json")
    print(f"staged in {stage}:")
    for p in sorted(stage.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    if args.dry_run:
        print("dry run: nothing uploaded. Review README.md in the staging folder.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(stage), commit_message="Upload Dictum dictation adapter")
    print(f"published: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
