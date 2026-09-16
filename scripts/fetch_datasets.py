#!/usr/bin/env python3
"""Download the human-authored eval datasets into evals/data/ (git-ignored).

Datasets (both human-written, neither AI-generated):
- Disfl-QA (Google Research, CC BY 4.0). Human annotators rewrote SQuAD
  questions to add spoken disfluencies: corrections ("no wait"), restarts.
  Pairs of disfluent -> original fluent question.
  https://github.com/google-research-datasets/Disfl-QA
- DisfluencySpeech (AMAAI Lab, 2024, Apache 2.0). ~5,000 utterances whose text
  comes from Switchboard telephone conversations, with human-annotated
  transcripts at increasing levels of cleanup. Only the text columns are
  downloaded, not the audio. https://huggingface.co/datasets/amaai-lab/DisfluencySpeech
- NL2Bash (Lin et al., LREC 2018, data MIT). ~10k English descriptions written
  by Bash programmers, paired with one-liners collected from sites such as
  Stack Overflow. https://github.com/TellinaTool/nl2bash

Usage: python scripts/fetch_datasets.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evals" / "data"

FILES = {
    "disflqa/dev.json": "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/dev.json",
    "disflqa/train.json": "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/train.json",
    "disflqa/test.json": "https://raw.githubusercontent.com/google-research-datasets/Disfl-QA/main/test.json",
    "nl2bash/all.nl": "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl",
    "nl2bash/all.cm": "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm",
}


def main() -> int:
    for rel, url in FILES.items():
        dest = DATA / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {rel} <- {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    fetch_disflspeech()
    print(f"done. data in {DATA}")
    return 0


def fetch_disflspeech() -> None:
    """Text columns of DisfluencySpeech -> evals/data/disflspeech/<split>.jsonl.

    Reads only the transcript columns from the Hub's parquet files with range
    requests, so the ~10 hours of audio is never downloaded.
    """
    import json
    out_dir = DATA / "disflspeech"
    if all((out_dir / f"{s}.jsonl").exists() for s in ("train", "validation", "test")):
        print("have  disflspeech/*.jsonl")
        return
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError:
        print("skip  disflspeech (pip install pyarrow huggingface_hub)")
        return
    fs = HfFileSystem()
    repo = "datasets/amaai-lab/DisfluencySpeech"
    files = fs.glob(f"{repo}/**/*.parquet")
    if not files:
        files = fs.glob(f"{repo}@refs%2Fconvert%2Fparquet/**/*.parquet")
    if not files:
        print("skip  disflspeech (no parquet files found on the Hub)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {"train": [], "validation": [], "test": []}
    for path in sorted(files):
        name = path.rsplit("/", 1)[-1].lower() + " " + path.lower()
        split = "validation" if ("valid" in name or "/dev" in name) else "test" if "test" in name else "train"
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            cols = [c for c in pf.schema_arrow.names if c.startswith("transcript")]
            table = pf.read(columns=cols)
        rows[split].extend(table.to_pylist())
        print(f"fetch disflspeech {split} <- {path.split('/', 3)[-1]} ({table.num_rows} rows)")
    for split, rs in rows.items():
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
