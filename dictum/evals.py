"""Refinement eval harness.

Suites
------
targeted  evals/cases/<mode>.yaml. Hand-written cases aimed at specific failure
          modes, scored by explicit checks. Written with an AI assistant, so
          used for regression testing, not headline numbers.
disflqa   Disfl-QA (human annotators). Disfluent question -> original question.
          Mode: dictation. Scored by word error rate against the original.
nl2bash   NL2Bash (human programmers). Description -> command.
          Mode: command. Scored by utility sequence match plus flag F1.

External suites are split into dev (tune on) and test (report only) with a
fixed seed. See evals/DATASETS.md.

This evaluates the refine stage alone. STT accuracy is `dictum bench`.
"""
from __future__ import annotations

import json
import math
import random
import re
import shlex
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
DATA_DIR = ROOT / "evals" / "data"
RESULTS_DIR = ROOT / "evals" / "results"
DIFFICULTIES = ("easy", "medium", "hard")
SUITES = ("targeted", "disflqa", "nl2bash")
SEED = 20260915
SIZES = {"disflqa": {"dev": 200, "test": 500}, "nl2bash": {"dev": 100, "test": 300}}
# NL2Bash is a Linux dataset and is executed in a Linux sandbox, so the model is
# told it is on Linux there. macOS behaviour is covered by the targeted suite.
SUITE_VARS = {"nl2bash": {"platform_hint": "This is Linux with bash and GNU coreutils; use GNU flags."}}

WER_PASS = 0.10         # disflqa: at most 1 word in 10 wrong vs the original question


@dataclass
class Case:
    id: str
    mode: str
    difficulty: str
    input: str
    reference: str = ""
    tags: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    dictionary: list | None = None
    scorer: str = "checks"         # checks | wer | command
    suite: str = "targeted"


# --------------------------------------------------------------------------- loading

def load_targeted(modes: list[str] | None = None) -> list[Case]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        mode = path.stem
        if modes and mode not in modes:
            continue
        for raw in yaml.safe_load(path.read_text()) or []:
            meta = ("id", "difficulty", "input", "reference", "tags", "dictionary")
            c = Case(id=raw["id"], mode=mode, difficulty=raw.get("difficulty", "medium"),
                     input=raw["input"], reference=raw.get("reference", ""), tags=raw.get("tags", []),
                     checks={k: raw[k] for k in raw if k not in meta}, dictionary=raw.get("dictionary"))
            if c.difficulty not in DIFFICULTIES:
                raise ValueError(f"{path.name}:{c.id}: difficulty must be one of {DIFFICULTIES}")
            cases.append(c)
    return cases


def load_cases(modes: list[str] | None = None) -> list[Case]:  # backwards compatible name
    return load_targeted(modes)


def _need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}. Run: python scripts/fetch_datasets.py")
    return path


def load_disflqa(split: str) -> list[Case]:
    data = json.loads(_need(DATA_DIR / "disflqa" / f"{split}.json").read_text())
    items = sorted(data.items())
    random.Random(SEED).shuffle(items)
    out = []
    for qid, v in items[: SIZES["disflqa"][split]]:
        out.append(Case(id=f"disflqa-{qid}", mode="dictation", difficulty="external", input=v["disfluent"],
                        reference=v["original"], scorer="wer", suite="disflqa", dictionary=[]))
    return out


def load_nl2bash(split: str) -> list[Case]:
    nl = _need(DATA_DIR / "nl2bash" / "all.nl").read_text().splitlines()
    cm = _need(DATA_DIR / "nl2bash" / "all.cm").read_text().splitlines()
    pairs = [(i, n.strip(), c.strip()) for i, (n, c) in enumerate(zip(nl, cm))
             if n.strip() and c.strip() and not n.startswith("(GNU specific)")]
    pairs = [(i, re.sub(r"^\((BSD|macOS) specific\)\s*", "", n), c) for i, n, c in pairs]
    random.Random(SEED).shuffle(pairs)
    exe_path = DATA_DIR / "nl2bash" / "executable.json"
    if exe_path.exists() and split in ("dev", "test"):
        # Keep only cases whose reference actually runs in the sandbox fixture
        # (see `dictum prep-nl2bash`). Order is preserved, so splits stay seeded.
        ok = set(json.loads(exe_path.read_text())["executable"])
        pairs = [p for p in pairs if p[0] in ok]
    n_dev, n_test = SIZES["nl2bash"]["dev"], SIZES["nl2bash"]["test"]
    # dev and test come from disjoint slices at the front; the rest is reserved as train.
    if exe_path.exists() and len(pairs) < n_dev + n_test:
        # Too few runnable cases for the nominal sizes: keep the 1:3 dev:test ratio.
        n_dev = len(pairs) // 4
        n_test = len(pairs) - n_dev
    chosen = pairs[:n_dev] if split == "dev" else pairs[n_dev:n_dev + n_test]
    return [Case(id=f"nl2bash-{i}", mode="command", difficulty="external", input=n, reference=c,
                 scorer="command", suite="nl2bash", dictionary=[]) for i, n, c in chosen]


def nl2bash_candidates() -> list[tuple[int, str, str]]:
    """All usable pairs in seeded order, before any executability filter."""
    nl = _need(DATA_DIR / "nl2bash" / "all.nl").read_text().splitlines()
    cm = _need(DATA_DIR / "nl2bash" / "all.cm").read_text().splitlines()
    pairs = [(i, n.strip(), c.strip()) for i, (n, c) in enumerate(zip(nl, cm))
             if n.strip() and c.strip() and not n.startswith("(GNU specific)")]
    random.Random(SEED).shuffle(pairs)
    return pairs


def load_suite(suite: str, split: str, modes: list[str] | None) -> list[Case]:
    if suite == "targeted":
        return load_targeted(modes)
    if suite == "disflqa":
        return load_disflqa(split)
    if suite == "nl2bash":
        return load_nl2bash(split)
    raise SystemExit(f"unknown suite '{suite}'. Choose from {SUITES}")


# --------------------------------------------------------------------------- scoring

def _words(s: str) -> list[str]:
    return re.findall(r"[\w'@./:-]+", s.lower())


def _norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower().replace("’", "'"))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _words(a), _words(b)).ratio() if a and b else 0.0


def wer(hyp: str, ref: str) -> float:
    h, r = _norm_words(hyp), _norm_words(ref)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


_LONG_SINGLE_DASH = {"find", "java", "ffmpeg", "xcodebuild", "security", "defaults", "osascript"}
_WRAPPERS = {"sudo", "xargs", "time", "nohup", "env", "exec", "command", "builtin", "nice"}


class _Backtick(list):
    pass


def command_parts(cmd: str) -> tuple[list[str], set[str]]:
    """Utilities used (top level and inside $( ) / backticks) and the set of flags.

    Quote-aware: a | inside a quoted regex is not a pipe. Wrappers like sudo
    and xargs are skipped so `xargs mv` counts as mv.
    """
    cmd = cmd.replace("\\;", " ").replace("$(", " ( ").replace("`", " ` ")
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars="|&;()")
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        toks = cmd.split()
    utils: list[str] = []
    flags: set[str] = set()

    def flush(seg: list[str]) -> None:
        t = [x for x in seg if not re.match(r"^[A-Za-z_]\w*=", x)]
        while t and t[0] in _WRAPPERS:
            t = t[1:]
            while t and t[0].startswith("-"):
                t = t[1:]
        if not t or t[0].startswith("-") or t[0] == "{}":
            return
        util = t[0].split("/")[-1]
        utils.append(util)
        for x in t[1:]:
            if re.match(r"^--?[A-Za-z]", x):
                x = x.split("=")[0]
                if re.match(r"^-[A-Za-z]{2,}$", x) and util not in _LONG_SINGLE_DASH:   # -la -> -l -a
                    flags.update(f"-{ch}" for ch in x[1:])
                else:
                    flags.add(x)

    stack: list[list[str]] = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            if len(stack) > 1:
                flush(stack.pop())
        elif t == "`":
            if len(stack) > 1 and isinstance(stack[-1], _Backtick):
                flush(stack.pop())
            else:
                stack.append(_Backtick())
        elif t and set(t) <= set("|&;"):
            flush(stack[-1])
            stack[-1].clear()
        else:
            stack[-1].append(t)
    while stack:
        flush(stack.pop())
    return utils, flags


def command_score(hyp: str, ref: str) -> dict:
    hu, hf = command_parts(hyp)
    ru, rf = command_parts(ref)
    if not hf and not rf:
        f1 = 1.0
    else:
        tp = len(hf & rf)
        p = tp / len(hf) if hf else 0.0
        r = tp / len(rf) if rf else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"util_match": set(hu) == set(ru), "flag_f1": f1}


def check(case: Case, output: str) -> list[str]:
    """Failure reasons for a case; empty list means pass."""
    out = output.strip()
    if not out:
        return ["empty output"]
    if case.scorer == "wer":
        w = wer(out, case.reference)
        fails = [f"wer {w:.2f} > {WER_PASS}"] if w > WER_PASS else []
        return fails
    if case.scorer == "command":
        s = command_score(out, case.reference)
        fails = []
        if not s["util_match"]:
            fails.append(f"utilities {sorted(set(command_parts(out)[0]))} != {sorted(set(command_parts(case.reference)[0]))}")
        if "\n" in out:
            fails.append("not single line")
        return fails

    c, fails, low = case.checks, [], out.lower()
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
    return fails


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- running

def run_eval(cfg: Config, modes: list[str] | None, models: list[str] | None, repeat: int, console,
             suite: str = "targeted", split: str = "dev", limit: int | None = None,
             backend: str = "ollama", as_mode: str | None = None) -> dict:
    from . import dictionary as dict_mod
    from .refine import create_refiner

    cases = load_suite(suite, split, modes)
    if modes:
        cases = [c for c in cases if c.mode in modes]
    if limit:
        cases = cases[:limit]
    if as_mode:
        cases = [replace(c, mode=as_mode) for c in cases]
    if not cases:
        raise SystemExit("no cases selected")

    rcfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.refine.items()}
    if backend == "ollama":
        rcfg.setdefault("ollama", {})
        rcfg["ollama"]["timeout_s"] = max(60, rcfg["ollama"].get("timeout_s", 15))  # model swaps can be slow
    elif models:
        raise SystemExit("--models only applies to the ollama backend")
    refiner = create_refiner(rcfg, cfg.dictionary, backend=backend)
    default_entries = refiner.entries

    by_mode: dict[str, list[Case]] = {}
    for c in cases:
        by_mode.setdefault(c.mode, []).append(c)

    rows = []
    for mode_name, mode_cases in by_mode.items():
        base = cfg.mode(mode_name)
        for model in (models or [refiner.model_for(base)]):
            mode = replace(base, model=model, vars={**base.vars, **SUITE_VARS.get(suite, {})})
            with console.status(f"warming {model} for {mode_name}..."):
                refiner.warm_up(mode)
            t_start = time.perf_counter()
            for i, case in enumerate(mode_cases, 1):
                for r in range(repeat):
                    eta = (time.perf_counter() - t_start) / max(1, i - 1) * (len(mode_cases) - i + 1) if i > 1 else 0
                    console.print(f"[dim]{suite}/{split} {mode_name} {model} {i}/{len(mode_cases)} "
                                  f"(~{eta / 60:.0f} min left)[/]" + " " * 10, end="\r")
                    tr = Transcript(text=case.input, engine="eval", latency_s=0.0, audio_s=0.0)
                    refiner.entries = dict_mod.parse(case.dictionary) if case.dictionary is not None else default_entries
                    notes = []
                    try:
                        res = refiner.refine(tr, mode)
                        out, lat, err, notes = res.text, res.latency_s, None, res.notes
                    except Exception as e:
                        out, lat, err = "", 0.0, f"{type(e).__name__}: {e}"
                    fails = check(case, out) if err is None else [err]
                    row = {"suite": suite, "split": split, "mode": mode_name, "model": model, "case": case.id,
                           "difficulty": case.difficulty, "tags": case.tags, "run": r,
                           "input": case.input, "reference": case.reference, "output": out,
                           "passed": not fails, "failures": fails, "notes": notes,
                           "latency_s": round(lat, 3), "similarity": round(similarity(out, case.reference), 3)}
                    if case.scorer == "wer":
                        row["wer"] = round(wer(out, case.reference), 3)
                    if case.scorer == "command":
                        row.update(command_score(out, case.reference))
                    rows.append(row)
    console.print(" " * 100, end="\r")
    return {"timestamp": datetime.now().isoformat(timespec="seconds"), "suite": suite, "split": split, "rows": rows}


def summarize(results: dict) -> list[dict]:
    rows = results["rows"]
    out = []
    for key in sorted({(r["mode"], r["model"]) for r in rows}):
        rs = [r for r in rows if (r["mode"], r["model"]) == key]
        lat = [r["latency_s"] for r in rs if r["latency_s"] > 0]
        k = sum(r["passed"] for r in rs)
        lo, hi = wilson(k, len(rs))
        s = {"suite": results.get("suite", "targeted"), "split": results.get("split", ""),
             "mode": key[0], "model": key[1], "n": len(rs), "pass": k / len(rs), "ci_low": lo, "ci_high": hi,
             "guard_trips": sum(any(n.startswith("guard") for n in r.get("notes", [])) for r in rs),
             "p50_s": float(np.percentile(lat, 50)) if lat else float("nan"),
             "p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
             "similarity": float(np.mean([r["similarity"] for r in rs if r["reference"]] or [float("nan")]))}
        if any("wer" in r for r in rs):
            s["mean_wer"] = float(np.mean([r["wer"] for r in rs if "wer" in r]))
        if any("flag_f1" in r for r in rs):
            s["flag_f1"] = float(np.mean([r["flag_f1"] for r in rs if "flag_f1" in r]))
        ex = [r for r in rs if r.get("exec_executable")]
        if ex:
            k_ex = sum(bool(r.get("exec_match")) for r in ex)
            s["exec_n"], s["exec_pass"] = len(ex), k_ex / len(ex)
            s["exec_ci_low"], s["exec_ci_high"] = wilson(k_ex, len(ex))
        for d in DIFFICULTIES:
            ds = [r for r in rs if r["difficulty"] == d]
            s[d] = (sum(r["passed"] for r in ds) / len(ds)) if ds else None
        out.append(s)
    return out


def save(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{results.get('suite', 'targeted')}-{results.get('split', '')}".rstrip("-")
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}.json"
    path.write_text(json.dumps(dict(results, summary=summarize(results)), indent=2))
    return path


def markdown_table(summary: list[dict]) -> str:
    lines = ["| suite | split | mode | model | n | pass | 95% CI | p50 s | p95 s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['suite']} | {s['split']} | {s['mode']} | {s['model']} | {s['n']} | {s['pass']:.0%} | "
                     f"{s['ci_low']:.0%} to {s['ci_high']:.0%} | {s['p50_s']:.2f} | {s['p95_s']:.2f} |")
    return "\n".join(lines)
