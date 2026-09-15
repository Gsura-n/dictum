"""Deterministic checks on LLM output. The model proposes, code disposes.

If a guard trips, the refiner falls back to the input transcript. Pasting the
user's own words unpolished is always better than pasting a poem they did not
ask for.
"""
from __future__ import annotations


def _n(s: str) -> int:
    return len(s.split())


def check(guard: dict, source: str, output: str) -> tuple[str, str | None]:
    """Returns (text_to_use, reason_or_None)."""
    if not guard:
        return output, None
    out = output
    if guard.get("single_line"):
        lines = [l for l in out.splitlines() if l.strip()]
        if len(lines) > 1:
            return source if guard.get("fallback_on_multiline") else lines[0], "multi-line output"
        out = lines[0] if lines else ""
    if not out.strip():
        return source, "empty output"
    n_in, n_out = _n(source), _n(out)
    if "max_ratio" in guard and n_out > max(guard["max_ratio"] * n_in, n_in + guard.get("slack_words", 8)):
        return source, f"output too long ({n_out} vs {n_in} words)"
    if "min_ratio" in guard and n_in >= 6 and n_out < guard["min_ratio"] * n_in:
        return source, f"output too short ({n_out} vs {n_in} words)"
    return out, None
