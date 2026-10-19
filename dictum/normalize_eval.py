"""Measure the deterministic normalize rules: what they fix and what they break.

normalize runs before the LLM and the output guards, so its mistakes reach the
user unchecked. Each case says which rule it targets and whether that rule
should fire. A "should not fire" case is people talking ABOUT punctuation, or an
ordinary word that collides with a trigger ("period of time", "new line of
products"). Output must equal `expected` exactly.

Per rule: recall = should-fire cases that pass; precision = TP / (TP + misfires),
where a misfire is a should-not-fire case that failed because that rule fired.
The mix is about half and half, so precision describes this mix, not the rate in
real speech. No models are involved; this is pure Python and cheap.

Attribution. A rule is credited with changing a text when switching its flag off
changes the output, or when it alone changes the text. The punctuation group has
one flag, so within it the rule is the one whose trigger phrase disappeared. Two
rules that make the same fix (restarts and phrase repeats on "X and X and X") are
both credited. A failing should-not-fire case where no rule is found is reported
as unattributed rather than guessed. Recall is by outcome: a should-fire case
passes if the text came out right.

Cases live in evals/normalize/{dev,test}.yaml. Tune against dev; score test only
when evaluating a fix.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import normalize as n
from .config import DEFAULT_PATH, ROOT
from .evals import RESULTS_DIR, wilson

CASES_DIR = ROOT / "evals" / "normalize"
SPLITS = ("dev", "test")
POSITIONS = ("", "mid", "end")

# rule name -> trigger phrase the rule consumes (punctuation group only)
_TRIGGERS = {
    "comma": r"\bcomma\b",
    "period": r"\bperiod\b",
    "full_stop": r"\bfull stop\b",
    "question_mark": r"\bquestion mark\b",
    "exclamation": r"\bexclamation (?:mark|point)\b",
    "semicolon": r"\bsemi ?colon\b",
    "colon": r"(?<!semi )\bcolon\b",
    "new_line": r"\b(?:new|next) line\b",
    "new_paragraph": r"\b(?:new|next) paragraph\b",
}
# rule name -> apply() flag, for the rules that have a flag of their own
_FLAG = {"repeated_words": "repeated_words", "restart": "restarts",
         "phrase_repeat": "phrase_repeats", "scratch_that": "scratch_that"}
RULES = ("email", "url", *_TRIGGERS, *_FLAG)


@dataclass
class Case:
    id: str
    rule: str
    fire: bool
    input: str
    expected: str
    position: str = ""      # comma, period, new_line: "mid" (inside a clause) or "end" (clause end)
    note: str = ""


def default_cfg() -> dict:
    """The committed default normalize settings; a local override must not change a measurement."""
    return yaml.safe_load(DEFAULT_PATH.read_text())["refine"]["normalize"]


def load(split: str) -> list[Case]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = CASES_DIR / f"{split}.yaml"
    out = []
    for raw in yaml.safe_load(path.read_text()) or []:
        c = Case(id=raw["id"], rule=raw["rule"], fire=raw["fire"], input=raw["input"], expected=raw["expected"],
                 position=raw.get("position", ""), note=raw.get("note", ""))
        if c.rule not in RULES:
            raise ValueError(f"{path.name}:{c.id}: unknown rule {c.rule!r}; known: {RULES}")
        if not isinstance(c.fire, bool):
            raise ValueError(f"{path.name}:{c.id}: fire must be true or false")
        if c.position not in POSITIONS:
            raise ValueError(f"{path.name}:{c.id}: position must be one of {POSITIONS}")
        out.append(c)
    return out


def run(text: str, cfg: dict) -> str:
    return n.join(n.apply(text, cfg))


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


_FLAGS = ("spoken_addresses", "repeated_words", "restarts", "phrase_repeats", "spoken_punctuation", "scratch_that")


def fired_rules(text: str, cfg: dict) -> list[str]:
    """Rules that change this text: a flag counts if switching it off changes the output, or if it alone
    (every other flag off) changes the text. The second test catches two rules that can each make the same
    fix, where switching either one off changes nothing."""
    all_off = {**cfg, **{f: False for f in _FLAGS}}
    full, bare = run(text, cfg), run(text, all_off)
    fired = []
    for flag in _FLAGS:
        off, solo = run(text, {**cfg, flag: False}), run(text, {**all_off, flag: True})
        if off != full:
            before, after = off, full
        elif solo != bare:
            before, after = bare, solo
        else:
            continue
        if flag == "spoken_addresses":
            fired.append("email" if after.count("@") > before.count("@") else "url")
        elif flag == "spoken_punctuation":
            fired += [r for r, pat in _TRIGGERS.items() if _count(pat, before) > _count(pat, after)]
        else:
            fired.append(next(r for r, f in _FLAG.items() if f == flag))
    return fired


def score(cases: list[Case], cfg: dict) -> list[dict]:
    rows = []
    for c in cases:
        out = run(c.input, cfg)
        rows.append({"id": c.id, "rule": c.rule, "fire": c.fire, "position": c.position, "input": c.input,
                     "expected": c.expected, "output": out, "passed": out == c.expected,
                     "fired": fired_rules(c.input, cfg)})
    return rows


def _ratio(k: int, total: int) -> tuple[float | None, list[float] | None]:
    if total == 0:
        return None, None
    lo, hi = wilson(k, total)
    return k / total, [lo, hi]


def _rule_stats(tp: int, fn: int, fp: int, n_fire: int, n_nofire: int) -> dict:
    precision, p_ci = _ratio(tp, tp + fp)
    recall, r_ci = _ratio(tp, tp + fn)
    return {"n_fire": n_fire, "n_nofire": n_nofire, "tp": tp, "fn": fn, "fp": fp,
            "precision": precision, "precision_ci": p_ci, "recall": recall, "recall_ci": r_ci}


def summarize(rows: list[dict]) -> dict:
    """Per-rule and pooled precision and recall with 95% Wilson intervals."""
    rules, T = {}, {"tp": 0, "fn": 0, "fp": 0}
    for rule in RULES:
        fire = [r for r in rows if r["rule"] == rule and r["fire"]]
        tp = sum(r["passed"] for r in fire)
        fn = len(fire) - tp
        # charged to the rule that fired, whatever the case's target was
        fp = sum(1 for r in rows if not r["fire"] and not r["passed"] and rule in r["fired"])
        n_nofire = sum(1 for r in rows if r["rule"] == rule and not r["fire"])
        rules[rule] = _rule_stats(tp, fn, fp, len(fire), n_nofire)
        T["tp"] += tp
        T["fn"] += fn
        T["fp"] += fp
    passed = sum(r["passed"] for r in rows)
    pass_rate, pass_ci = _ratio(passed, len(rows))
    overall = _rule_stats(T["tp"], T["fn"], T["fp"], T["tp"] + T["fn"], sum(1 for r in rows if not r["fire"]))
    overall.update({
        "n": len(rows), "passed": passed, "pass_rate": pass_rate, "pass_ci": pass_ci,
        # a should-not-fire case that failed with no rule found: cannot be charged to anyone
        "unattributed_failures": sum(1 for r in rows if not r["fire"] and not r["passed"] and not r["fired"]),
        # should-fire cases that also tripped a different rule; listed, not scored as misfires
        "collateral_in_fire_cases": sum(1 for r in rows if r["fire"] and set(r["fired"]) - {r["rule"]}),
    })
    return {"rules": rules, "overall": overall}


def save(split: str, rows: list[dict], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-normalize-{split}.json"
    path.write_text(json.dumps({"suite": "normalize", "split": split, "rows": rows, "summary": summary}, indent=2))
    return path
