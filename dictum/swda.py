"""Switchboard Dialog Act Corpus (SwDA) -> (disfluent, clean) pairs, by rule.

SwDA transcripts carry the Meteer et al. (1995) disfluency markup:
  {F uh}       filled pause            -> removed in clean
  {D you know} discourse marker        -> removed in clean
  {E I mean}   explicit editing term   -> removed in clean
  {C and}      coordinating conjunction-> kept (braces stripped)
  {A ...}      aside                   -> kept
  [ X + Y ]    repair: X replaced by Y -> clean keeps Y only (nested, and
                                          "[ X + ]" drops X entirely)
  <Laughter>   non-speech              -> removed from both sides
  /  -/  --    unit boundaries / interruptions

The disfluent side keeps every spoken word; the clean side applies the edits
above. Both come from human annotation; nothing is model-generated.
Consecutive slash-units from the same speaker turn are joined, so inputs look
like multi-sentence dictation rather than single fragments.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .disflspeech import tidy_target

_NONSPEECH = re.compile(r"<+[^<>]*>+|\(\([^)]*\)\)|#")
_BRACE = re.compile(r"\{([FDECAK])\s+([^{}]*)\}")


def _find_matching(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _top_level_plus(s: str) -> int:
    depth = 0
    for j, ch in enumerate(s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "+" and depth == 0:
            return j
    return -1


def _repairs(s: str, clean: bool) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "[":
            j = _find_matching(s, i)
            if j == -1:                       # unbalanced: drop the bracket
                i += 1
                continue
            inner = s[i + 1:j]
            k = _top_level_plus(inner)
            if k == -1:
                out.append(_repairs(inner, clean))
            else:
                reparandum, repair = inner[:k], inner[k + 1:]
                out.append(_repairs(repair, clean) if clean else
                           _repairs(reparandum, clean) + " " + _repairs(repair, clean))
            i = j + 1
        elif s[i] in "]+":
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _braces(s: str, clean: bool) -> str:
    if clean:
        # "different, {F uh, } areas" -> "different areas": a removed filler takes its commas with it
        s = re.sub(r",\s*\{[FDE]\s+[^{}]*,\s*\}", " ", s)
    prev = None
    while prev != s:                          # innermost first, for nested braces
        prev = s
        s = _BRACE.sub(lambda m: (" " if clean and m.group(1) in "FDE" else f" {m.group(2)} "), s)
    return s


def _tidy(s: str) -> str:
    s = s.replace("-/", " ").replace("/", " ").replace("--", " ")
    s = re.sub(r"\b\w+-(?=[\s,.]|$)", " ", s)  # partial words "nucle-"
    s = re.sub(r"(?<=\s)-(?=\s)|,\s*-\s", " ", s)  # stray dashes from cut-offs
    s = re.sub(r"\s+([,.?!])", r"\1", s)
    s = re.sub(r"([,.?!])(\s*[,.?!])+", r"\1", s)
    s = re.sub(r"^[\s,.]+", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def disfluent_and_clean(markup: str) -> tuple[str, str]:
    base = _NONSPEECH.sub(" ", markup)
    disfl = _tidy(_repairs(_braces(base, clean=False), clean=False))
    clean = _tidy(_repairs(_braces(base, clean=True), clean=True))
    return disfl, clean


def load_turns(root: Path, max_units: int = 3) -> list[dict]:
    """Groups consecutive slash-units of one speaker's turn (up to max_units) into one example."""
    rows = []
    for f in sorted(root.glob("**/*utt.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append((r["conversation_no"], r["caller"], int(r["utterance_index"]), r["text"]))
    turns: list[dict] = []
    cur_key, buf = None, []

    def flush():
        for k in range(0, len(buf), max_units):
            chunk = " ".join(buf[k:k + max_units])
            tail = chunk.rstrip()
            if tail.endswith(("--", "-/", "-")) or chunk.strip().startswith("--"):
                continue                      # interrupted or continued across turns
            src, tgt = disfluent_and_clean(chunk)
            if len(src.split()) >= 5 and len(tgt.split()) >= 3:
                turns.append({"conv": cur_key[0], "src": src, "tgt": tidy_target(tgt),
                              "edited": src.lower() != tgt.lower()})

    for conv, caller, uidx, text in rows:
        key = (conv, caller, uidx)
        if key != cur_key:
            if buf:
                flush()
            cur_key, buf = key, []
        buf.append(text)
    if buf:
        flush()
    return turns
