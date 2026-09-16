"""Shared refine flow for LLM backends.

  transcript
    -> dictionary respellings (code)
    -> normalize: addresses, repeats, spoken punctuation, paragraph split (code)
    -> LLM per segment
    -> output guards per segment (code)
    -> rejoin with the original line/paragraph breaks
"""
from __future__ import annotations

import time
from typing import Callable

from .. import dictionary as dict_mod
from .. import normalize
from ..config import Mode
from ..types import Refined, Transcript
from . import guards
from .ollama_refiner_utils import clean_output


def refine_with(generate: Callable[[Mode, str], str], name: str, transcript: Transcript, mode: Mode,
                entries, normalize_cfg: dict | None) -> Refined:
    t0 = time.perf_counter()
    notes: list[str] = []
    source = dict_mod.apply(transcript.text, entries)
    if source != transcript.text:
        notes.append("dictionary applied")

    if mode.name.startswith("command"):
        segments = [(source, "")]          # commands: no punctuation or paragraph rules
    else:
        segments = normalize.apply(source, normalize_cfg)
        if normalize.join(segments) != source:
            notes.append("normalized")

    out = []
    for seg, brk in segments:
        if not seg.strip():
            out.append((seg, brk))
            continue
        raw = clean_output(generate(mode, seg))
        raw = dict_mod.apply(raw, entries)
        text, reason = guards.check(mode.guard, seg, raw)
        if reason:
            notes.append(f"guard: {reason}; used transcript")
        out.append((text, brk))
    return Refined(text=normalize.join(out), backend=name, mode=mode.name,
                   latency_s=time.perf_counter() - t0, source=transcript, notes=notes)
