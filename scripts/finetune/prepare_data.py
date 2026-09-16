#!/usr/bin/env python3
"""Build LoRA training data for the dictation cleaner.

Output: evals/data/finetune/dictation/{train,valid}.jsonl (mlx-lm chat format)
and report.json describing the mix.

Sources (training splits only; every dev/test split stays untouched):
- Disfl-QA train: typed disfluent questions -> fluent question.
- DisfluencySpeech train: Switchboard statements, transcript_a -> transcript_c.
- SwDA: Switchboard turns with human disfluency markup, converted by rule.

Mix is capped per source (see --caps) because an unbalanced mix taught the
model "every output is a question" when Disfl-QA dominated.

Leak protection:
- exact-text matches against Disfl-QA and DisfluencySpeech dev/test are dropped
- SwDA and DisfluencySpeech both come from Switchboard, so any SwDA turn that
  shares an 8-word sequence with a DisfluencySpeech dev/test sentence is dropped
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dictum.config import Config  # noqa: E402

SEED = 20260915
DATA = ROOT / "evals" / "data"
OUT = DATA / "finetune" / "dictation"
N_VALID = 400


def norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def shingles(t: str, n: int = 8) -> set[str]:
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="disflqa=4000,disflspeech=4500,swda=16000",
                    help="max rows per source before identity rows")
    ap.add_argument("--identity", type=float, default=0.2, help="extra already-clean rows, as a fraction")
    ap.add_argument("--swda-unedited", type=float, default=0.15,
                    help="share of SwDA rows allowed to need no edits (teaches leaving clean speech alone)")
    args = ap.parse_args()
    caps = {k: int(v) for k, v in (x.split("=") for x in args.caps.split(","))}

    system = Config.load().mode("dictation_ft").prompt.strip()
    rng = random.Random(SEED)
    dropped = Counter()

    # ---- held-out text
    load_q = lambda s: json.loads((DATA / "disflqa" / f"{s}.json").read_text())  # noqa: E731
    held = {norm(t) for s in ("dev", "test") for v in load_q(s).values() for t in (v["original"], v["disfluent"])}
    ds_dir = DATA / "disflspeech"
    have_ds = (ds_dir / "train.jsonl").exists()
    held_shingles: set[str] = set()
    if have_ds:
        from dictum.disflspeech import load_pairs
        for s in ("validation", "test"):
            for pair in load_pairs(ds_dir / f"{s}.jsonl"):
                for t in pair:
                    held.add(norm(t))
                    held_shingles |= shingles(t)

    pools: dict[str, list[tuple[str, str]]] = {}

    q = []
    for v in (v for _, v in sorted(load_q("train").items())):
        src, tgt = v["disfluent"].strip(), v["original"].strip()
        if not src or not tgt:
            dropped["disflqa empty"] += 1
        elif norm(src) in held or norm(tgt) in held:
            dropped["disflqa overlaps eval"] += 1
        else:
            q.append((src, tgt))
    pools["disflqa"] = q

    if have_ds:
        ds = []
        for src, tgt in load_pairs(ds_dir / "train.jsonl"):
            if norm(src) in held or norm(tgt) in held:
                dropped["disflspeech overlaps eval"] += 1
            else:
                ds.append((src, tgt))
        pools["disflspeech"] = ds

    swda_root = DATA / "swda"
    if any(swda_root.glob("**/*utt.csv")):
        from dictum.swda import load_turns
        edited, unedited = [], []
        for t in load_turns(swda_root):
            if norm(t["src"]) in held or norm(t["tgt"]) in held or (shingles(t["src"]) & held_shingles):
                dropped["swda overlaps disflspeech eval"] += 1
                continue
            (edited if t["edited"] else unedited).append((t["src"], t["tgt"]))
        rng.shuffle(edited)
        rng.shuffle(unedited)
        cap = caps.get("swda", len(edited))
        n_un = int(cap * args.swda_unedited)
        pools["swda"] = edited[: cap - n_un] + unedited[:n_un]
    else:
        print("WARNING: SwDA not found; run scripts/fetch_datasets.py")

    rows = []
    for name, pool in pools.items():
        rng.shuffle(pool)
        pool = pool[: caps.get(name, len(pool))]
        rows += [(name, s, t) for s, t in pool]
        for _, t in rng.sample(pool, int(len(pool) * args.identity)):
            rows.append((f"{name}-identity", t, t))
    rng.shuffle(rows)

    def chat(src, tgt):
        return {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"<transcript>{src}</transcript>"},
                             {"role": "assistant", "content": tgt}]}

    OUT.mkdir(parents=True, exist_ok=True)
    for split, part in (("valid", rows[:N_VALID]), ("train", rows[N_VALID:])):
        with open(OUT / f"{split}.jsonl", "w") as f:
            for _, s, t in part:
                f.write(json.dumps(chat(s, t)) + "\n")

    mix = Counter(n for n, _, _ in rows)
    report = {"total_rows": len(rows), "train": len(rows) - N_VALID, "valid": N_VALID, "caps": caps,
              "mix": dict(mix), "dropped": dict(dropped),
              "samples": {n: [{"in": s, "out": t} for m, s, t in rows if m == n][:3] for n in mix}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"rows: {len(rows)} -> train {len(rows) - N_VALID}, valid {N_VALID}")
    print(f"mix: {dict(mix)}")
    print(f"dropped: {dict(dropped)}")
    print(f"estimated training time at micro-batch 4, 0.32 it/s: {(len(rows) - N_VALID) / 4 / 0.32 / 3600:.1f} h")
    for n in ("swda",):
        for smp in report["samples"].get(n, []):
            print(f"  sample {n}: {smp['in']!r} -> {smp['out']!r}")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
