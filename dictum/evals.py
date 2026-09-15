"""Refinement eval harness.

Cases live in evals/cases/<mode>.yaml. Each case is a raw transcript (what the
STT stage would hand us), a reference output, and a set of checks. A case
passes only if every check passes. The reference is not used for pass/fail,
because there are many acceptable ways to write the same sentence; it is used
for a similarity score and for humans reading failures.

Checks (all optional):
  include        substrings that must appear (case-insensitive)
  include_exact  substrings that must appear with exact casing (names, APIs)
  exclude        words/phrases that must NOT appear (case-insensitive, word bounded)
  regex          patterns that must match (case-insensitive)
  not_regex      patterns that must not match
  single_line    output must be one line
  min_len_ratio  output words / reference words lower bound (catches summarising)
  max_len_ratio  upper bound (catches the model adding or answering things)

This evaluates the refine stage in isolation. STT accuracy is measured
separately by `dictum bench`, so a failure here is always the LLM's.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml

from .config import ROOT, Config
from .types import Transcript

CASES_DIR = ROOT / "evals" / "cases"
RESULTS_DIR = ROOT / "evals" / "results"
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class Case:
    id: str
    mode: str
    difficulty: str
    input: str
    reference: str = ""
    tags: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def load_cases(modes: list[str] | None = None) -> list[Case]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        mode = path.stem
        if modes and mode not in modes:
            continue
        for raw in yaml.safe_load(path.read_text()) or []:
            checks = {k: raw[k] for k in raw if k not in ("id", "difficulty", "input", "reference", "tags")}
            c = Case(id=raw["id"], mode=mode, difficulty=raw.get("difficulty", "medium"),
                     input=raw["input"], reference=raw.get("reference", ""), tags=raw.get("tags", []),
                     checks=checks)
            if c.difficulty not in DIFFICULTIES:
                raise ValueError(f"{path.name}:{c.id}: difficulty must be one of {DIFFICULTIES}")
            cases.append(c)
    return cases


def _words(s: str) -> list[str]:
    return re.findall(r"[\w'@./:-]+", s.lower())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _words(a), _words(b)).ratio() if a and b else 0.0


def check(case: Case, output: str) -> list[str]:
    """Returns a list of failure reasons; empty means pass."""
    c, out, fails = case.checks, output.strip(), []
    low = out.lower()
    for s in c.get("include", []):
        if s.lower() not in low:
            fails.append(f"missing '{s}'")
    for s in c.get("include_exact", []):
        if s not in out:
            fails.append(f"missing exact '{s}'")
    for s in c.get("exclude", []):
        if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", out, re.IGNORECASE):
            fails.append(f"contains '{s}'")
    for p in c.get("regex", []):
        if not re.search(p, out, re.IGNORECASE | re.MULTILINE):
            fails.append(f"no match /{p}/")
    for p in c.get("not_regex", []):
        if re.search(p, out, re.IGNORECASE | re.MULTILINE):
            fails.append(f"forbidden /{p}/")
    if c.get("single_line") and "\n" in out:
        fails.append("not single line")
    if case.reference and ("min_len_ratio" in c or "max_len_ratio" in c):
        ratio = len(_words(out)) / max(1, len(_words(case.reference)))
        if ratio < c.get("min_len_ratio", 0):
            fails.append(f"too short (x{ratio:.2f})")
        if ratio > c.get("max_len_ratio", 99):
            fails.append(f"too long (x{ratio:.2f})")
    if not out:
        fails.append("empty output")
    return fails


def run_eval(cfg: Config, modes: list[str] | None, models: list[str] | None, repeat: int, console) -> dict:
    from .refine.ollama_refiner import OllamaRefiner

    cases = load_cases(modes)
    if not cases:
        raise SystemExit(f"no cases found in {CASES_DIR}")
    refiner = OllamaRefiner(dictionary=cfg.dictionary, **cfg.refine.get("ollama", {}))

    rows = []
    by_mode: dict[str, list[Case]] = {}
    for c in cases:
        by_mode.setdefault(c.mode, []).append(c)

    for mode_name, mode_cases in by_mode.items():
        base = cfg.mode(mode_name)
        for model in (models or [refiner.model_for(base)]):
            mode = replace(base, model=model)
            with console.status(f"warming {model} for {mode_name}..."):
                refiner.warm_up(mode)
            for i, case in enumerate(mode_cases, 1):
                for r in range(repeat):
                    console.print(f"[dim]{mode_name} {model} {i}/{len(mode_cases)} {case.id}[/]", end="\r")
                    tr = Transcript(text=case.input, engine="eval", latency_s=0.0, audio_s=0.0)
                    try:
                        res = refiner.refine(tr, mode)
                        out, lat, err = res.text, res.latency_s, None
                    except Exception as e:
                        out, lat, err = "", 0.0, f"{type(e).__name__}: {e}"
                    fails = check(case, out) if err is None else [err]
                    rows.append({"mode": mode_name, "model": model, "case": case.id,
                                 "difficulty": case.difficulty, "tags": case.tags, "run": r,
                                 "input": case.input, "reference": case.reference, "output": out,
                                 "passed": not fails, "failures": fails, "latency_s": round(lat, 3),
                                 "similarity": round(similarity(out, case.reference), 3)})
    console.print(" " * 80, end="\r")
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "rows": rows}


def summarize(results: dict) -> list[dict]:
    rows = results["rows"]
    out = []
    for key in sorted({(r["mode"], r["model"]) for r in rows}):
        rs = [r for r in rows if (r["mode"], r["model"]) == key]
        lat = [r["latency_s"] for r in rs if r["latency_s"] > 0]
        s = {"mode": key[0], "model": key[1], "n": len(rs),
             "pass": sum(r["passed"] for r in rs) / len(rs),
             "p50_s": float(np.percentile(lat, 50)) if lat else float("nan"),
             "p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
             "similarity": float(np.mean([r["similarity"] for r in rs if r["reference"]] or [float("nan")]))}
        for d in DIFFICULTIES:
            ds = [r for r in rs if r["difficulty"] == d]
            s[d] = (sum(r["passed"] for r in ds) / len(ds)) if ds else None
        out.append(s)
    return out


def save(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    results = dict(results, summary=summarize(results))
    path.write_text(json.dumps(results, indent=2))
    return path


def markdown_table(summary: list[dict]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"
    lines = ["| mode | model | pass | easy | medium | hard | p50 s | p95 s |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['mode']} | {s['model']} | {pct(s['pass'])} | {pct(s['easy'])} | "
                     f"{pct(s['medium'])} | {pct(s['hard'])} | {s['p50_s']:.2f} | {s['p95_s']:.2f} |")
    return "\n".join(lines)
