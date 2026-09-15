#!/usr/bin/env python3
"""Download the human-authored eval datasets into evals/data/ (git-ignored).

Datasets (both human-written, neither AI-generated):
- Disfl-QA (Google Research, CC BY 4.0). Human annotators rewrote SQuAD
  questions to add spoken disfluencies: corrections ("no wait"), restarts.
  Pairs of disfluent -> original fluent question.
  https://github.com/google-research-datasets/Disfl-QA
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
    print(f"done. data in {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
