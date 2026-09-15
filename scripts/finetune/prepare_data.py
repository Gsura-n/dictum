#!/usr/bin/env python3
"""Build LoRA training data for the dictation cleaner from Disfl-QA train.

Output: evals/data/finetune/dictation/{train,valid}.jsonl in mlx-lm chat format.

Choices (and why):
- Source is ONLY the Disfl-QA train split. dev/test stay untouched for eval.
- Any train item whose original or disfluent text also appears in dev or test
  is dropped, so no test question leaks in through a duplicate.
- ~25% extra "identity" rows (fluent original -> itself) teach the model to
  leave clean text alone instead of rewriting everything it sees.
- The system prompt is the short `dictation_ft` prompt from config. The model
  learns the behaviour from data, so it does not need the long rulebook and
  seven examples at inference, which also makes it faster.
- valid is carved from train (not dev) so dev remains an honest eval set.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dictum.config import Config  # noqa: E402

SEED = 20260915
DATA = ROOT / "evals" / "data"
OUT = DATA / "finetune" / "dictation"


def wrap(t: str) -> str:
    return f"<transcript>{t}</transcript>"


def row(system: str, src: str, tgt: str) -> dict:
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": wrap(src)},
                         {"role": "assistant", "content": tgt}]}


def main() -> int:
    system = Config.load().mode("dictation_ft").prompt.strip()
    load = lambda s: json.loads((DATA / "disflqa" / f"{s}.json").read_text())  # noqa: E731
    train, dev, test = load("train"), load("dev"), load("test")
    held_out = {t.strip().lower() for d in (dev, test) for v in d.values() for t in (v["original"], v["disfluent"])}

    items = [v for _, v in sorted(train.items())
             if v["original"].strip().lower() not in held_out and v["disfluent"].strip().lower() not in held_out
             and v["original"].strip() and v["disfluent"].strip()]
    dropped = len(train) - len(items)
    rng = random.Random(SEED)
    rng.shuffle(items)

    rows = [row(system, v["disfluent"].strip(), v["original"].strip()) for v in items]
    identity = [row(system, v["original"].strip(), v["original"].strip()) for v in rng.sample(items, len(items) // 4)]
    rows += identity
    rng.shuffle(rows)

    n_valid = 300
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    print(f"disfl-qa train items: {len(train)}, dropped for dev/test overlap or empty: {dropped}")
    print(f"rows: {len(rows)} ({len(items)} corrections + {len(identity)} identity) -> "
          f"train {len(rows) - n_valid}, valid {n_valid}")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
