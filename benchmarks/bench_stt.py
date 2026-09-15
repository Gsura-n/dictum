"""Benchmark STT engines on a folder of WAV clips.

For each clip_NNN.wav, an optional clip_NNN.txt holds the reference transcript;
when present, word error rate (WER) is computed with jiwer.
Results go to <out>/<timestamp>.json and are printed as a table.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from dictum.config import Config
from dictum.stt import create_engine
from dictum.types import AudioClip


def _load_clip(path: Path) -> AudioClip:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        # Simple resample via linear interpolation; fine for benchmarking speech.
        n = int(len(samples) * 16000 / sr)
        samples = np.interp(np.linspace(0, len(samples), n, endpoint=False), np.arange(len(samples)), samples)
        sr = 16000
    return AudioClip(samples=samples.astype(np.float32), sample_rate=sr)


def run_benchmark(samples: Path, engines: list[str], out: Path, console: Console) -> None:
    clips = sorted(samples.glob("*.wav"))
    if not clips:
        console.print(f"[red]no .wav files in {samples}[/]. Record some with: dictum listen --save {samples}")
        return
    try:
        import jiwer
    except ImportError:
        jiwer = None
        console.print("[yellow]jiwer not installed; skipping WER (pip install 'dictum[dev]')[/]")

    cfg = Config.load()
    results = []
    for name in engines:
        _, ecfg = cfg.stt_engine_config(name)
        engine = create_engine(name, **ecfg)
        t0 = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - t0
        for wav in clips:
            clip = _load_clip(wav)
            tr = engine.transcribe(clip)
            ref_path = wav.with_suffix(".txt")
            wer = None
            if jiwer and ref_path.exists():
                wer = jiwer.wer(ref_path.read_text().strip().lower(), tr.text.lower())
            results.append({"engine": name, "model": ecfg.get("model"), "clip": wav.name,
                            "audio_s": round(clip.duration_s, 2), "latency_s": round(tr.latency_s, 3),
                            "rtf": round(tr.rtf, 3), "wer": wer, "load_s": round(load_s, 2), "text": tr.text})

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(results, indent=2))

    t = Table("engine", "clips", "load s", "mean latency s", "mean rtf", "mean wer", title="STT benchmark")
    for name in engines:
        rows = [r for r in results if r["engine"] == name]
        wers = [r["wer"] for r in rows if r["wer"] is not None]
        t.add_row(name, str(len(rows)), f"{rows[0]['load_s']:.1f}",
                  f"{np.mean([r['latency_s'] for r in rows]):.2f}",
                  f"{np.mean([r['rtf'] for r in rows]):.3f}",
                  f"{np.mean(wers):.3f}" if wers else "n/a")
    console.print(t)
    console.print(f"[dim]details: {path}[/]")
