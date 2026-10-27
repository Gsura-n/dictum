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


import re

def strip_preamble(text: str, source: str) -> str:
    prefixes = ("Here is", "Here's", "Sure", "Certainly", "Okay", "Below is")
    if text.startswith(prefixes):
        lines = text.split("\n")
        if len(lines) > 1 and lines[0].strip().endswith(":"):
            if any(line.strip() for line in lines[1:]):
                # Count words in preamble line that appear in source
                preamble_words = re.findall(r'\b\w+\b', lines[0].lower())
                if preamble_words:
                    source_words = set(re.findall(r'\b\w+\b', source.lower()))
                    match_count = sum(1 for w in preamble_words if w in source_words)

                    # If fewer than half the words are in source, it's generated preamble
                    if match_count < len(preamble_words) / 2:
                        if not lines[1].strip():
                            return "\n".join(lines[2:])
                        else:
                            return "\n".join(lines[1:])
    return text

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
    n_chunks = 0
    for seg, brk in segments:
        if not seg.strip():
            out.append((seg, brk))
            continue
        parts = []
        for piece in normalize.chunk(seg, mode.chunk_words):
            n_chunks += 1
            raw = clean_output(generate(mode, piece))
            raw = strip_preamble(raw, piece)
            raw = dict_mod.apply(raw, entries)
            text, reason = guards.check(mode.guard, piece, raw)
            if reason:
                notes.append(f"guard: {reason}; used transcript")
            parts.append(text)
        out.append((" ".join(parts), brk))
    if n_chunks > len([s for s, _ in segments if s.strip()]):
        notes.append(f"refined in {n_chunks} chunks")
    return Refined(text=normalize.join(out), backend=name, mode=mode.name,
                   latency_s=time.perf_counter() - t0, source=transcript, notes=notes)


def refine_levels(refiner, cfg, transcript: Transcript, mode: Mode) -> Refined:
    """Run a mode's `pre_mode` first and refine its output.

    Cleanup levels stack rather than compete: the fine-tuned model does the
    faithful pass, because that is the best thing we have at disfluency, and the
    level above edits its output into prose. Each stage keeps its own guards, so
    a bad second stage falls back to a good first stage instead of to raw speech.
    """
    first = refiner.refine(transcript, cfg.mode(mode.pre_mode))
    staged = Transcript(text=first.text, engine=transcript.engine,
                        latency_s=transcript.latency_s, audio_s=transcript.audio_s)
    refined = refiner.refine(staged, mode)
    refined.source = transcript
    refined.latency_s += first.latency_s
    refined.notes = [f"{mode.pre_mode}: {n}" for n in first.notes] + refined.notes
    return refined
