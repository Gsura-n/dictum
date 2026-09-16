"""Audio for eval cases, so the whole pipeline (speech -> STT -> refine) can be scored.

Two sources:
- real: DisfluencySpeech ships human recordings of every utterance. Fetched by
  scripts/fetch_datasets.py --audio into evals/data/disflspeech/audio/<split>/.
- tts: any other suite is spoken with the macOS `say` command straight to a WAV
  file (nothing plays aloud), rotating across installed English voices. Synthetic
  voices pronounce "uh" and "um" cleanly, so this measures how STT copes with
  disfluent wording, not with real human hesitation.

Audio is cached, so generating it once is enough.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .evals import DATA_DIR, Case

TTS_DIR = DATA_DIR / "tts"
PREFERRED_VOICES = ["Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa", "Fred", "Victoria"]


def english_voices() -> list[str]:
    if sys.platform != "darwin" or shutil.which("say") is None:
        raise SystemExit("TTS audio needs the macOS `say` command")
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}(en[_-][A-Z]{2})\b", line)
        if m and "(" not in m.group(1):        # skip novelty and duplicated enhanced variants
            voices.append(m.group(1).strip())
    preferred = [v for v in PREFERRED_VOICES if v in voices]
    return preferred or voices[:6]


def tts_file(text: str, voice: str, suite: str) -> Path:
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    path = TTS_DIR / suite / f"{key}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text],
                       check=True, capture_output=True)
    return path


def attach_audio(cases: list[Case], suite: str, split: str, source: str, console=None) -> list[Case]:
    """Returns cases with .audio set; cases without audio are dropped (and counted)."""
    if source == "real":
        if suite != "disflspeech":
            raise SystemExit("real audio is only available for --suite disflspeech (use --audio tts)")
        manifest = DATA_DIR / "disflspeech" / "audio" / f"{'validation' if split == 'dev' else 'test'}.jsonl"
        if not manifest.exists():
            raise SystemExit(f"missing {manifest}. Run: python scripts/fetch_datasets.py --audio")
        from .disflspeech import strip_markup
        by_text = {}
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            by_text.setdefault(strip_markup(r["transcript_a"]), r["audio"])
        out = [replace(c, audio=str(DATA_DIR / "disflspeech" / "audio" / by_text[c.input]))
               for c in cases if c.input in by_text]
    elif source == "tts":
        voices = english_voices()
        out = []
        for i, c in enumerate(cases):
            if console:
                console.print(f"[dim]synthesizing audio {i + 1}/{len(cases)}[/]" + " " * 10, end="\r")
            out.append(replace(c, audio=str(tts_file(c.input, voices[i % len(voices)], suite)),
                               tags=[*c.tags, f"voice:{voices[i % len(voices)]}"]))
    else:
        raise SystemExit("--audio must be real or tts")
    if console and len(out) < len(cases):
        console.print(f"[yellow]{len(cases) - len(out)} cases had no audio and were skipped[/]")
    return out
