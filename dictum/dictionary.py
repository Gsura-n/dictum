"""Personal dictionary.

An entry is either a plain word ("Gauttam") or a word with the ways the STT
engine tends to mishear it:

    - word: Gauttam
      sounds_like: [gautam, gowtham]

sounds_like variants are replaced deterministically BEFORE the LLM sees the
text. Asking a small model to fix spellings is unreliable (it ignored the
instruction in evals, and sometimes spread the name into unrelated words like
email addresses). A word-bounded substitution is exact and costs nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Entry:
    word: str
    sounds_like: list[str] = field(default_factory=list)


def parse(raw) -> list[Entry]:
    out = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(Entry(item))
        elif isinstance(item, dict) and "word" in item:
            out.append(Entry(item["word"], [str(s) for s in item.get("sounds_like", []) or []]))
        else:
            raise ValueError(f"bad dictionary entry: {item!r}")
    return out


def apply(text: str, entries: list[Entry]) -> str:
    # Longest variants first so "use effect hook" wins over "use effect".
    # The word itself is also a variant, so "gauttam" gets its casing fixed.
    pairs = sorted(((v, e.word) for e in entries for v in [*e.sounds_like, e.word]), key=lambda p: -len(p[0]))
    for variant, word in pairs:
        pattern = r"(?<![\w@./-])" + r"\s+".join(map(re.escape, variant.split())) + r"(?![\w@-])"
        text = re.sub(pattern, word, text, flags=re.IGNORECASE)
    return text


def words(entries: list[Entry]) -> list[str]:
    return [e.word for e in entries]
