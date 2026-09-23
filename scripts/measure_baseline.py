#!/usr/bin/env python3
"""Phase 0 baseline: memory and latency of the path the hotkeys actually use.

Runs the real pipeline (the same Pipeline.process the hotkeys call: STT, then
refine) on recorded human speech from the DisfluencySpeech dev split, with a
no-op injector so nothing is pasted. Consecutive recordings are joined into
utterances of about --seconds each, because the latency target is stated for a
10 s utterance.

Reports:
  - model load time
  - memory: process RSS, MLX peak, macOS phys_footprint (includes GPU buffers),
    and `ollama ps` when a mode on this path uses Ollama
  - p50 / p95 of STT, refine and total time. Total is what you wait after
    releasing the key, minus stopping the mic and the paste itself (tens of ms)

Usage, on the Mac inside .venv:
  python scripts/fetch_datasets.py --audio      # once, if evals/data/disflspeech/audio is missing
  python scripts/measure_baseline.py            # default hotkey mode, 40 utterances of ~10 s
  python scripts/measure_baseline.py --mode dictation --n 40

Close other heavy apps for a clean number, or leave your usual browser and IDE
open for a realistic one, and say which in the notes (--note).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dictum.config import Config  # noqa: E402
from dictum.pipeline import Pipeline  # noqa: E402
from dictum.types import AudioClip  # noqa: E402

AUDIO = ROOT / "evals" / "data" / "disflspeech" / "audio"
SR = 16000


class NullInjector:
    name = "null"

    def inject(self, text: str) -> None:
        pass


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def machine() -> dict:
    return {
        "model": sh("sysctl", "-n", "hw.model"),
        "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1) if sys.platform == "darwin" else None,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "commit": sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"),
        "dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")),
    }


def memory() -> dict:
    pid = os.getpid()
    out = {"rss_mb": round(int(sh("ps", "-o", "rss=", "-p", str(pid)) or 0) / 1024)}
    # ru_maxrss is bytes on macOS, KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_mb"] = round(peak / 2**20 if sys.platform == "darwin" else peak / 1024)
    try:
        import mlx.core as mx
        get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        out["mlx_active_mb"] = round(get_active() / 2**20)
        out["mlx_peak_mb"] = round(get_peak() / 2**20)
    except Exception:
        pass
    # phys_footprint is what Activity Monitor calls Memory; it counts Metal buffers
    fp = sh("footprint", str(pid))
    out["footprint"] = [line.strip() for line in fp.splitlines() if "footprint" in line.lower()][:3] or fp[:300]
    return out


def utterances(seconds: float, n: int) -> list[tuple[AudioClip, str]]:
    import soundfile as sf
    import soxr

    manifest = AUDIO / "validation.jsonl"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}. Run: python scripts/fetch_datasets.py --audio")
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    out, buf, words = [], [], []
    for line in manifest.read_text().splitlines():
        r = json.loads(line)
        x, sr = sf.read(AUDIO / r["audio"], dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = soxr.resample(x, sr, SR).astype(np.float32)
        if len(x) / SR > 1.3 * seconds:
            continue                                  # one recording already far too long
        have = sum(len(b) for b in buf) / SR
        if have + len(x) / SR > 1.3 * seconds:       # this recording would overshoot
            if have < 0.6 * seconds:
                continue                              # too short to flush yet; try the next one
            out.append((AudioClip(np.concatenate(buf[:-1]), SR), " ".join(words)))
            buf, words = [], []
        buf += [x, gap]
        words.append(r["transcript_a"])
        if sum(len(b) for b in buf) / SR >= seconds:
            out.append((AudioClip(np.concatenate(buf[:-1]), SR), " ".join(words)))
            buf, words = [], []
        if len(out) >= n:
            break
    return out[:n]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", help="mode to measure (default: what Right Option uses outside terminals)")
    ap.add_argument("--n", type=int, default=40, help="number of utterances")
    ap.add_argument("--seconds", type=float, default=10.0, help="target utterance length")
    ap.add_argument("--note", default="", help="e.g. 'browser and VS Code open'")
    ap.add_argument("--out", type=Path, help="result file (default evals/results/baseline-<time>.json)")
    a = ap.parse_args()

    cfg = Config.load()
    mode = a.mode or cfg.raw.get("apps", {}).get("default", "dictation")
    warm = cfg.raw.get("hotkeys", {}).get("warm_modes") or [mode]
    print(f"mode {mode}; warming {warm} like `dictum run` does")

    clips = utterances(a.seconds, a.n)
    mem0 = memory()
    # Built by hand rather than with Pipeline.from_config: that opens the mic,
    # and this script feeds recorded audio, so it must run with no input device.
    from dictum.refine import create_refiner
    from dictum.stt import create_engine
    name, ecfg = cfg.stt_engine_config(None)
    pipe = Pipeline(capture=None, stt=create_engine(name, **ecfg),
                    refiner=create_refiner(cfg.refine, cfg.dictionary), injector=NullInjector(), cfg=cfg)
    t0 = time.perf_counter()
    load = pipe.warm_up(modes=list(warm))
    load_s = time.perf_counter() - t0
    mem_warm = memory()
    print(f"loaded in {load_s:.1f}s; rss {mem_warm['rss_mb']} MB, mlx {mem_warm.get('mlx_active_mb', '?')} MB")

    rows = []
    for i, (clip, ref) in enumerate(clips, 1):
        refined, t = pipe.process(clip, mode)
        rows.append({"audio_s": round(t.audio_s, 2), "stt_s": round(t.stt_s, 3), "refine_s": round(t.refine_s, 3),
                     "total_s": round(t.total_s, 3), "backend": refined.backend, "notes": refined.notes,
                     "output": refined.text, "reference_transcript": ref})
        print(f"  {i:>3}/{len(clips)}  {t}", flush=True)

    mem_end = memory()
    ollama = sh("ollama", "ps") if any("ollama" in str(r["backend"]) for r in rows) else None
    col = lambda k: [r[k] for r in rows]  # noqa: E731
    summary = {k: {"p50": round(pct(col(k), 50), 3), "p95": round(pct(col(k), 95), 3)}
               for k in ("stt_s", "refine_s", "total_s")}
    summary["audio_s_mean"] = round(float(np.mean(col("audio_s"))), 1)
    summary["failed"] = sum(str(r["backend"]).startswith("failed") for r in rows)

    result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "machine": machine(), "mode": mode,
              "warm_modes": list(warm), "note": a.note, "load_s": round(load_s, 1), "load": load,
              "memory": {"before_load": mem0, "after_load": mem_warm, "end": mem_end, "ollama_ps": ollama},
              "summary": summary, "rows": rows}
    out = a.out or ROOT / "evals" / "results" / f"baseline-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    m = result["machine"]
    print(f"\n{m['chip']} {m['memory_gb']} GB, commit {m['commit']}{' (dirty)' if m['dirty'] else ''}")
    print(f"mode {mode}, {len(rows)} utterances of ~{summary['audio_s_mean']} s, {summary['failed']} failed")
    for k in ("stt_s", "refine_s", "total_s"):
        print(f"  {k[:-2]:>7}  p50 {summary[k]['p50']:.2f} s   p95 {summary[k]['p95']:.2f} s")
    print(f"  memory  rss {mem_end['rss_mb']} MB (peak {mem_end['peak_rss_mb']}), "
          f"mlx peak {mem_end.get('mlx_peak_mb', '?')} MB")
    print(f"  footprint {mem_end['footprint']}")
    if ollama:
        print(ollama)
    print(f"saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
