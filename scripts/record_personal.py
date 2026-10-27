#!/usr/bin/env python3
"""Guided recorder for a personal clip suite. Everything it writes goes under ~/dictum-personal/, never the repo.

Input, ~/dictum-personal/clips.yaml:
  clips:
    - {id: R01, kind: read, mode: dictation_ft, say: "what to read", expect: "what it should come out as"}
    - {id: C01, kind: cue,  mode: dictation_ft, cue: "tell me about your morning"}
Output:
  audio/<id>.wav        16 kHz mono
  transcripts.jsonl     one line per kept clip: id, kind, mode, transcript (raw STT), audio_s, timestamp

For each clip: Enter starts recording, Enter stops. Then Enter keeps it, r redoes it, s skips it, q quits.
It resumes at the first clip with no audio, so you can stop and come back. A clip is written only when kept.
`--from ID` starts at that clip and re-records it even if it has audio (the old take and its line are replaced).

Usage, on the Mac inside .venv:
  python scripts/record_personal.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
import yaml  # noqa: E402

from dictum.capture.sounddevice_capture import resample  # noqa: E402
from dictum.types import AudioClip  # noqa: E402

DEFAULT_DIR = Path.home() / "dictum-personal"
SR = 16000
KINDS = ("read", "cue")
_ID = re.compile(r"[A-Za-z0-9_.-]+")


def ensure_outside_repo(path: Path) -> None:
    p = Path(path).expanduser().resolve()
    if p == ROOT or ROOT in p.parents:
        raise ValueError(f"{p} is inside the repo; personal clips must stay outside it")


def load_clips(path: Path) -> list[dict]:
    clips = (yaml.safe_load(Path(path).read_text()) or {}).get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"{path}: expected a non-empty list under 'clips'")
    seen = set()
    for c in clips:
        cid = str(c.get("id", ""))
        if not _ID.fullmatch(cid) or cid.startswith("."):
            raise ValueError(f"{path}: bad id {cid!r} (letters, digits, _ - . only)")
        if cid in seen:
            raise ValueError(f"{path}: duplicate id {cid}")
        seen.add(cid)
        if c.get("kind") not in KINDS:
            raise ValueError(f"{path}:{cid}: kind must be one of {KINDS}")
        if not c.get("mode"):
            raise ValueError(f"{path}:{cid}: mode is required")
        for need in (("say", "expect") if c["kind"] == "read" else ("cue",)):
            if not c.get(need):
                raise ValueError(f"{path}:{cid}: a {c['kind']} clip needs {need}")
    return clips


def wav_path(root: Path, cid: str) -> Path:
    return Path(root) / "audio" / f"{cid}.wav"


def first_missing(clips: list[dict], root: Path) -> int | None:
    return next((i for i, c in enumerate(clips) if not wav_path(root, c["id"]).exists()), None)


def _save(root: Path, c: dict, samples, text: str) -> None:
    wav = wav_path(root, c["id"])
    wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav.with_name(wav.name + ".tmp")
    sf.write(str(tmp), samples, SR, subtype="PCM_16", format="WAV")
    os.replace(tmp, wav)
    row = {"id": c["id"], "kind": c["kind"], "mode": c["mode"], "transcript": text,
           "audio_s": round(len(samples) / SR, 2), "timestamp": datetime.now().isoformat(timespec="seconds")}
    log = Path(root) / "transcripts.jsonl"
    old = [x for x in (log.read_text().splitlines() if log.exists() else []) if json.loads(x)["id"] != c["id"]]
    tmp = log.with_name(log.name + ".tmp")
    tmp.write_text("\n".join(old + [json.dumps(row)]) + "\n")
    os.replace(tmp, log)


def _prompt(ask, text: str, allowed: tuple[str, ...]) -> str:
    """One of the allowed answers ('' is Enter). Ctrl-D quits."""
    while True:
        try:
            a = ask(text).strip().lower()
        except EOFError:
            return "q"
        if a in allowed:
            return a
        print("  press " + ", ".join(k or "Enter" for k in allowed))


def _one(c: dict, n: int, total: int, root: Path, capture, stt, ask, show) -> str:
    """'kept', 'skipped' or 'quit'."""
    show(f"\n[{n}/{total}] {c['id']}   (mode {c['mode']})")
    show("READ:" if c["kind"] == "read" else "SPEAK FREELY:")
    show(f"  {c['say'] if c['kind'] == 'read' else c['cue']}")
    while True:
        key = _prompt(ask, "Enter = start recording, s = skip, q = quit: ", ("", "s", "q"))
        if key in ("q", "s"):
            return {"q": "quit", "s": "skipped"}[key]
        capture.start()
        if _prompt(ask, "  recording... Enter = stop: ", ("",)) != "":
            capture.stop()                     # Ctrl-D while recording: drop the take
            return "quit"
        clip = capture.stop()
        samples = resample(clip.samples, clip.sample_rate, SR)
        text = stt.transcribe(AudioClip(samples=samples, sample_rate=SR)).text
        show(f"  {len(samples) / SR:.1f} s heard: {text!r}")
        key = _prompt(ask, "  Enter = keep, r = redo, s = skip, q = quit: ", ("", "r", "s", "q"))
        if key == "":
            _save(root, c, samples, text)
            return "kept"
        if key != "r":
            return {"q": "quit", "s": "skipped"}[key]


def record_session(clips, root, capture, stt, ask=input, show=print, start_id: str | None = None) -> str:
    """'done' when it reaches the end, 'quit' when stopped early."""
    root = Path(root)
    ensure_outside_repo(root)
    ids = [c["id"] for c in clips]
    if start_id is not None:
        if start_id not in ids:
            raise ValueError(f"no clip with id {start_id}")
        start = ids.index(start_id)
    else:
        start = first_missing(clips, root)
        if start is None:
            show("Every clip already has audio. Use --from ID to redo one.")
            return "done"
    for n in range(start, len(clips)):
        c = clips[n]
        if start_id is None and wav_path(root, c["id"]).exists():
            continue                           # recorded in an earlier session
        if _one(c, n + 1, len(clips), root, capture, stt, ask, show) == "quit":
            return "quit"
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="folder with clips.yaml; output goes here too")
    ap.add_argument("--from", dest="start_id", help="start at this clip id and re-record it")
    ap.add_argument("--engine", help="STT engine (default: the one in the config)")
    ap.add_argument("--device", type=int, help="input device index (default: the config's)")
    a = ap.parse_args()
    root = a.dir.expanduser()
    ensure_outside_repo(root)
    clips = load_clips(root / "clips.yaml")

    # Same wiring as Pipeline.from_config, without the refiner and injector.
    from dictum.capture import SoundDeviceCapture
    from dictum.config import Config
    from dictum.stt import create_engine
    cfg = Config.load()
    au = cfg.audio
    name, ecfg = cfg.stt_engine_config(a.engine)
    capture = SoundDeviceCapture(au.get("sample_rate", SR), au.get("channels", 1),
                                 a.device if a.device is not None else au.get("device"))
    stt = create_engine(name, **ecfg)
    print(f"loading {name}...", flush=True)
    stt.load()
    print(f"mic: {capture.device_name}; {len(clips)} clips in {root}")
    try:
        status = record_session(clips, root, capture, stt, start_id=a.start_id)
    except KeyboardInterrupt:
        status = "quit"
        print("\nstopped")
    left = [c["id"] for c in clips if not wav_path(root, c["id"]).exists()]
    print(f"{status}: {len(clips) - len(left)}/{len(clips)} clips have audio" + (f"; next {left[0]}" if left else ""))


if __name__ == "__main__":
    main()
