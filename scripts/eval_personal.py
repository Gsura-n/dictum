#!/usr/bin/env python3
"""Evaluate ~/dictum-personal/{clips,references}.yaml and audio/<id>.wav locally.

--only ID[,ID...] and --mode filter clips; JSON stays outside the repo.
Latency excludes failures and separates <=15s clips for the 10s target; STT WER uses available transcripts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import yaml  # noqa: E402

from dictum.config import Config  # noqa: E402
from dictum.evals import WER_PASS, command_score, normalize_numbers, similarity, wer, wilson  # noqa: E402
from dictum.pipeline import Pipeline  # noqa: E402
from dictum.types import AudioClip  # noqa: E402
from scripts.measure_baseline import NullInjector, pct  # noqa: E402
from scripts.record_personal import DEFAULT_DIR, SR, ensure_outside_repo, wav_path  # noqa: E402

MODES = ("dictation_ft", "command", "academic")
GROUPS = "PETCNXAQ"


def load_suite(root: Path, only: str | None = None, mode: str | None = None) -> list[dict]:
    for path in (root, root / "clips.yaml", root / "references.yaml"):
        ensure_outside_repo(path)
    data = yaml.safe_load((root / "clips.yaml").read_text())
    refs = yaml.safe_load((root / "references.yaml").read_text())
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list) or not data["clips"]:
        raise ValueError("clips.yaml needs a non-empty list under clips")
    if not isinstance(refs, dict):
        raise ValueError("references.yaml needs a map of id to said/expect")
    seen, selected, wanted = set(), [], set(only.split(",")) if only is not None else None
    for c in data["clips"]:
        cid = c.get("id") if isinstance(c, dict) else None
        if not isinstance(cid, str) or not re.fullmatch(r"[PETCNXAQ][A-Za-z0-9_.-]*", cid) or cid in seen:
            raise ValueError("invalid or duplicate clip id; expected group letter P/E/T/C/N/X/A/Q")
        seen.add(cid)
        if c.get("kind") not in ("read", "cue") or c.get("mode") not in MODES:
            raise ValueError(f"{cid}: invalid kind or mode")
        if (wanted is not None and cid not in wanted) or (mode and c["mode"] != mode):
            continue
        ref = refs.get(cid)
        if not isinstance(ref, dict) or not isinstance(ref.get("said"), str):
            raise ValueError(f"{cid}: missing reference or non-string said")
        expect = ref.get("expect")
        valid = isinstance(expect, str) if c["mode"] != "command" else (
            isinstance(expect, list) and bool(expect) and all(isinstance(x, str) and x.strip() for x in expect))
        if not valid:
            raise ValueError(f"{cid}: expect must be text, or a non-empty command list for command mode")
        selected.append({"id": cid, "kind": c["kind"], "mode": c["mode"], "said": ref["said"], "expect": expect})
    if wanted is not None and wanted - seen:
        raise ValueError("--only contains unknown IDs")
    if not selected:
        raise ValueError("no clips selected")
    return selected


def build_pipeline(cfg: Config) -> Pipeline:
    from dictum.refine import create_refiner
    from dictum.stt import create_engine
    cfg.raw["history"] = {**cfg.raw.get("history", {}), "enabled": False}
    name, ecfg = cfg.stt_engine_config(None)
    return Pipeline(capture=None, stt=create_engine(name, **ecfg),
                    refiner=create_refiner(cfg.refine, cfg.dictionary), injector=NullInjector(), cfg=cfg)


def model_info(pipe, modes) -> dict:
    info, pending = {}, list(modes)
    while pending:
        name = pending.pop()
        if name in info:
            continue
        try:
            mode = pipe.cfg.mode(name)
            pending.extend(m for m in (mode.pre_mode, mode.fallback_mode) if m)
            backend = pipe.refiner.backend_for(mode) if hasattr(pipe.refiner, "backend_for") else pipe.refiner
            info[name] = {"backend": backend.name,
                          "model": backend.model_for(mode) if hasattr(backend, "model_for") else None,
                          "adapter": getattr(backend, "adapter_path", None),
                          "pre_mode": mode.pre_mode, "fallback_mode": mode.fallback_mode,
                          "stt": {"engine": pipe.cfg.stt_engine_config(None)[0], "model": getattr(pipe.stt, "model_id", None)}}
        except Exception as e:
            info[name] = {"error": f"{type(e).__name__}: {e}"}
    return info


def evaluate(clips, root: Path, pipe=None) -> tuple[list[dict], dict]:
    """One pipeline call per clip; WER costs O(reference words * hypothesis words)."""
    ensure_outside_repo(root)
    modes = list(dict.fromkeys(c["mode"] for c in clips))
    startup_error = None
    try:
        pipe = pipe or build_pipeline(Config.load())
        warm = list(dict.fromkeys((pipe.cfg.raw.get("hotkeys", {}).get("warm_modes") or []) + modes))
        pipe.warm_up(modes=warm)
    except Exception as e:
        startup_error = f"{type(e).__name__}: {e}"
    models, rows = (model_info(pipe, modes) if pipe else {m: {"error": startup_error} for m in modes}), []
    for mode, info in models.items():
        print(f"mode {mode}: {json.dumps(info)}")
    print("id       mode           STT WER   paste   total seconds")
    for c in clips:
        row = dict(c, graded=(c["kind"] != "cue" and c["mode"] != "academic"), passed=False, error=None, raw=None, output=None, audio_s=None,
                   stt_wer=None, stt_s=None, refine_s=None, total_s=None)
        start = None
        try:
            path = wav_path(root, c["id"])
            ensure_outside_repo(path)
            samples, sr = sf.read(path, dtype="float32")
            if sr != SR or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
                raise ValueError("audio must be non-empty finite 16 kHz mono")
            clip = AudioClip(samples, sr)
            row["audio_s"] = clip.duration_s
            if startup_error:
                raise RuntimeError(startup_error)
            start = time.perf_counter()
            result, timing = pipe.process(clip, c["mode"])
            row.update(raw=result.source.text, output=result.text, backend=result.backend,
                       output_mode=result.mode, notes=result.notes, stt_s=timing.stt_s,
                       refine_s=timing.refine_s, total_s=timing.total_s)
            if result.backend.startswith("failed"):
                row["error"] = result.backend
                # Pipeline returns zero refine time on fallback; retain the elapsed wait.
                row["refine_s"] = max(0, timing.total_s - timing.stt_s)
            row["stt_wer"] = wer(normalize_numbers(row["raw"]), normalize_numbers(c["said"]))
            out, expect = result.text, c["expect"]
            if c["mode"] == "command":
                passed = False
                for cmd in expect:
                    if out.strip() == cmd:
                        passed = True
                        break
                    score = command_score(out, cmd)
                    if score["util_match"] and score["flag_f1"] == 1.0:
                        passed = True
                        break
                row["passed"] = passed
                if not row["passed"]:
                    row["command_score"] = command_score(out, expect[0])
            else:
                row.update(paste_wer=wer(out, expect), paste_similarity=similarity(out, expect))
                row["passed"] = row["paste_wer"] <= WER_PASS and (
                    "\n" not in expect or out.count("\n") == expect.count("\n"))
            row["passed"] = row["passed"] and row["error"] is None
        except Exception as e:
            row.update(passed=False, error=f"{type(e).__name__}: {e}")
            if start is not None and row["total_s"] is None:
                row["stt_s"] = row["total_s"] = time.perf_counter() - start

        if not row["graded"]:
            row["passed"] = False

        status = 'pass' if row['passed'] else 'FAIL'
        if not row["graded"]:
            similarity_val = row.get("paste_similarity")
            status = f"-   {fmt(similarity_val)}"
        print(f"{row['id']:<8} {row['mode']:<14} {fmt(row['stt_wer']):>8}   {status:<5}   {fmt(row['total_s'])}")
        sys.stdout.flush()

        rows.append(row)
    return rows, models


def summarize(rows: list[dict]) -> dict:
    groups = {g: [r for r in rows if r["id"][0] == g] for g in GROUPS}
    groups["overall"] = rows
    summary = {}
    for group, rs in groups.items():
        graded_rs = [r for r in rs if r.get("graded", True)]
        ungraded_rs = [r for r in rs if not r.get("graded", True)]

        n, k = len(graded_rs), sum(r["passed"] for r in graded_rs)
        stt = [r["stt_wer"] for r in rs if r["stt_wer"] is not None]

        ungraded_sims = [r["paste_similarity"] for r in ungraded_rs if "paste_similarity" in r and r["paste_similarity"] is not None]

        s = dict(n=n, failed=sum(r["error"] is not None for r in rs), stt_n=len(stt),
                 mean_stt_wer=float(np.mean(stt)) if stt else None,
                 paste_pass_rate=k/n if n else None, paste_ci95=list(wilson(k, n)) if n else [None, None],
                 ungraded_n=len(ungraded_rs),
                 mean_paste_similarity=float(np.mean(ungraded_sims)) if ungraded_sims else None)
        for label, subset in (("latency", rs), ("latency_le15s", [r for r in rs if r["audio_s"] is not None and r["audio_s"] <= 15])):
            good = [r for r in subset if r["error"] is None]
            s[label] = {key: {f"p{p}": pct([r[key] for r in good], p) if good else None for p in (50, 95)}
                        for key in ("stt_s", "refine_s", "total_s")}
            s[label]["n"] = len(good)
        summary[group] = s
    return summary


def fmt(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def report(rows, summary, models):
    print("id       mode           STT WER   paste   total seconds")
    for r in rows:
        status = 'pass' if r['passed'] else 'FAIL'
        if not r["graded"]:
            similarity_val = r.get("paste_similarity")
            status = f"-   {fmt(similarity_val)}"
        print(f"{r['id']:<8} {r['mode']:<14} {fmt(r['stt_wer']):>8}   {status:<5}   {fmt(r['total_s'])}")
    print("Failures count against paste rate; latency excludes failed rows. STT WER uses available transcripts.")
    for group, s in summary.items():
        lo, hi = s["paste_ci95"]
        ungraded_text = ""
        if s.get("ungraded_n", 0) > 0:
            ungraded_text = f" mean similarity={fmt(s['mean_paste_similarity'])} (ungraded n={s['ungraded_n']})"

        print(f"{group}: n={s['n']} failed={s['failed']} mean STT WER={fmt(s['mean_stt_wer'])} "
              f"paste={fmt(s['paste_pass_rate'], '.1%')} Wilson 95% CI [{fmt(lo, '.1%')}, {fmt(hi, '.1%')}]{ungraded_text}")
        for key, label in (("latency", "all"), ("latency_le15s", "<=15s (10s target)")):
            values = "  ".join(f"{stage} p50={fmt(s[key][stage]['p50'])} p95={fmt(s[key][stage]['p95'])}"
                               for stage in ("stt_s", "refine_s", "total_s"))
            print(f"  {label}, n={s[key]['n']}: {values}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--only", help="comma-separated clip IDs")
    ap.add_argument("--mode", choices=MODES, help="only clips recorded in this mode")
    a = ap.parse_args()
    root = a.dir.expanduser().resolve()
    out = root / "results" / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    ensure_outside_repo(out)
    clips = load_suite(root, a.only, a.mode)
    rows, models = evaluate(clips, root)
    summary = summarize(rows)
    report(rows, summary, models)
    out.parent.mkdir(parents=True, exist_ok=True)
    ensure_outside_repo(out)
    with out.open("x") as f:
        json.dump(dict(timestamp=datetime.now().isoformat(), models=models, rows=rows, summary=summary), f, indent=2, allow_nan=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
