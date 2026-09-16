"""DisfluencySpeech text normalisation, shared by eval loading and training prep.

Columns: transcript_a (non-speech events removed; fillers, repairs and false
starts still present) and transcript_c (fillers, discourse markers, editing
terms and false starts removed). We use a -> c.

Switchboard-style annotation marks ({F uh}, [ ... + ... ], /) are stripped
from both sides if present, keeping the words. The target gets deterministic
sentence casing and a final period when it has none, so the model is not
taught to drop punctuation; nothing here is model-generated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_BRACE_TAG = re.compile(r"\{[A-Z]\s+")
_MARKUP = re.compile(r"[\[\]\{\}+/#]|<[^>]*>|\(\([^)]*\)\)")


def strip_markup(t: str) -> str:
    t = _BRACE_TAG.sub("", t or "")
    t = _MARKUP.sub(" ", t)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)          # ", ," left where a filler was removed
    t = re.sub(r",\s*([.?!])", r"\1", t)         # "though,." -> "though."
    return re.sub(r"\s+", " ", t).strip(" -,")


def tidy_target(t: str) -> str:
    t = strip_markup(t)
    if not t:
        return t
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'(m|ve|ll|d)\b", lambda m: "I'" + m.group(1), t)
    t = t[0].upper() + t[1:]
    t = re.sub(r"([.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    if t[-1] not in ".?!":
        t += "."
    return t


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            src, tgt = strip_markup(r.get("transcript_a", "")), tidy_target(r.get("transcript_c", ""))
            if len(src.split()) >= 3 and len(tgt.split()) >= 2:
                pairs.append((src, tgt))
    return pairs
