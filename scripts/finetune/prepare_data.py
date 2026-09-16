#!/usr/bin/env python3
"""Build LoRA training data for the dictation cleaner.

Output: evals/data/finetune/dictation/{train,valid}.jsonl in mlx-lm chat format,
plus report.json describing the mix.

Sources (training splits only; dev/test of every suite stay untouched):
- Disfl-QA train (7,182): typed disfluent questions -> fluent question.
- DisfluencySpeech train (~4,500): Switchboard conversational statements,
  transcript_a -> transcript_c. Adds statements, fillers and false starts,
  so the model does not learn "every output is a question".

Choices (and why):
- Any training row whose source or target text also appears in a dev or test
  set of either suite is dropped, so no eval item leaks in through a duplicate.
- ~25% extra identity rows (already-clean text -> itself) teach the model to
  leave clean text alone.
- The system prompt is the short `dictation_ft` prompt from config.
- valid is carved from the training mix, never from dev.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dictum.config import Config  # noqa: E402

SEED = 20260915
DATA = ROOT / "evals" / "data"
OUT = DATA / "finetune" / "dictation"
IDENTITY_FRACTION = 0.25
N_VALID = 300


def wrap(t: str) -> str:
    return f"<transcript>{t}</transcript>"


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def main() -> int:
    system = Config.load().mode("dictation_ft").prompt.strip()
    rng = random.Random(SEED)

    # ---- held-out text from every eval split, for leak filtering
    load_q = lambda s: json.loads((DATA / "disflqa" / f"{s}.json").read_text())  # noqa: E731
    held_out = {norm(t) for s in ("dev", "test") for v in load_q(s).values() for t in (v["original"], v["disfluent"])}

    ds_dir = DATA / "disflspeech"
    have_ds = all((ds_dir / f"{s}.jsonl").exists() for s in ("train", "validation", "test"))
    if have_ds:
        from dictum.disflspeech import load_pairs
        for s in ("validation", "test"):
            held_out |= {norm(t) for pair in load_pairs(ds_dir / f"{s}.jsonl") for t in pair}

    # ---- sources: (source_name, disfluent, clean)
    pairs: list[tuple[str, str, str]] = []
    dropped = Counter()
    for v in (v for _, v in sorted(load_q("train").items())):
        src, tgt = v["disfluent"].strip(), v["original"].strip()
        if not src or not tgt:
            dropped["disflqa empty"] += 1
        elif norm(src) in held_out or norm(tgt) in held_out:
            dropped["disflqa overlaps eval"] += 1
        else:
            pairs.append(("disflqa", src, tgt))
    if have_ds:
        for src, tgt in load_pairs(ds_dir / "train.jsonl"):
            if norm(src) in held_out or norm(tgt) in held_out:
                dropped["disflspeech overlaps eval"] += 1
            else:
                pairs.append(("disflspeech", src, tgt))
    else:
        print("WARNING: DisfluencySpeech not found; training on Disfl-QA only. Run scripts/fetch_datasets.py")

    rows = [(name, src, tgt) for name, src, tgt in pairs]
    by_source = Counter(name for name, _, _ in pairs)
    for name in by_source:
        pool = [p for p in pairs if p[0] == name]
        for _, _, tgt in rng.sample(pool, int(len(pool) * IDENTITY_FRACTION)):
            rows.append((f"{name}-identity", tgt, tgt))
    rng.shuffle(rows)

    def as_chat(src, tgt):
        return {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": wrap(src)},
                             {"role": "assistant", "content": tgt}]}

    OUT.mkdir(parents=True, exist_ok=True)
    for split, part in (("valid", rows[:N_VALID]), ("train", rows[N_VALID:])):
        with open(OUT / f"{split}.jsonl", "w") as f:
            for _, src, tgt in part:
                f.write(json.dumps(as_chat(src, tgt)) + "\n")

    mix = Counter(name for name, _, _ in rows)
    report = {"total_rows": len(rows), "train": len(rows) - N_VALID, "valid": N_VALID,
              "mix": dict(mix), "dropped": dict(dropped),
              "samples": {name: [{"in": s, "out": t} for n, s, t in rows if n == name][:3] for name in mix}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"rows: {len(rows)} -> train {len(rows) - N_VALID}, valid {N_VALID}")
    print(f"mix: {dict(mix)}")
    print(f"dropped: {dict(dropped)}")
    for name in ("disflspeech",):
        for smp in report["samples"].get(name, []):
            print(f"  sample {name}: {smp['in']!r} -> {smp['out']!r}")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
