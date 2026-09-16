"""Deterministic text rules applied before any LLM sees a transcript.

Anything that follows a fixed rule is done here, exactly, instead of hoping a
model does it: spoken punctuation ("comma", "new paragraph"), spoken email
addresses and URLs ("j patel at outlook dot com"), and stuttered repeats
("the the"). The fine-tuned model never saw these in training, and the
prompted models did them unreliably.

Paragraph and line breaks are returned as separate segments so each segment is
refined on its own and the breaks survive (small models flatten newlines).
"""
from __future__ import annotations

import re

TLDS = r"(?:com|org|net|io|edu|gov|co|ai|dev|app|me|us|uk|in|ca|de|info|biz|xyz)"
_W = r"[A-Za-z0-9]+"

# "period" is a real noun too ("trial period"); only treat it as punctuation
# when the previous word is not one that commonly precedes the noun.
_PERIOD_NOUN_BEFORE = {"the", "a", "this", "that", "trial", "grace", "time", "waiting", "probation",
                       "same", "each", "every", "long", "short", "first", "last", "next", "free",
                       "notice", "billing", "reporting", "cooling", "honeymoon", "transition", "holding"}

_SIMPLE = [
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation (?:mark|point)\b", "!"),
    (r"\bsemi ?colon\b", ";"),
    (r"\bcomma\b", ","),
    (r"\bfull stop\b", "."),
]
_BREAK = re.compile(r"\s*\b(new paragraph|next paragraph|new line|next line)\b[\s,.]*", re.IGNORECASE)

# words that may legitimately repeat
_ALLOWED_REPEATS = {"had", "that", "bye", "ha", "no", "very", "so"}


def _period(m: re.Match) -> str:
    prev = (m.group(1) or "").lower()
    return m.group(0) if prev in _PERIOD_NOUN_BEFORE else f"{m.group(1)}."


def spoken_punctuation(text: str) -> str:
    for pat, rep in _SIMPLE:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)\s+period\b", _period, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)\s+colon\b(?!\s*(?:cancer|polyp))", r"\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    return text


def _join_spoken(tokens: str) -> str:
    """'gsura dash n' -> 'gsura-n', 'first dot last' -> 'first.last'."""
    t = re.sub(r"\s+dash\s+|\s+hyphen\s+", "-", tokens, flags=re.IGNORECASE)
    t = re.sub(r"\s+underscore\s+", "_", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+dot\s+", ".", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", t).lower()


_STOP_LOCAL = {"at", "me", "is", "to", "email", "mail", "reach", "contact", "address", "my", "or", "and",
               "the", "on", "send", "it", "us", "him", "her", "them", "you", "your", "write",
               "look", "go", "visit", "see", "check", "find", "open", "found", "posted", "online", "live",
               "hosted", "available", "site", "website", "page", "repo", "link",
               "met", "meet", "be", "am", "are", "was", "were", "oh", "point", "here", "there", "him"}


def spoken_addresses(text: str) -> str:
    # URL: word(s) "dot" tld [ "slash" segment ]*
    seg = rf"{_W}(?:\s+(?:dash|hyphen|underscore|dot)\s+{_W})*"
    url = re.compile(rf"\b({_W}(?:\s+dot\s+{_W})*\s+dot\s+{TLDS})((?:\s+slash\s+{seg})*)\b", re.IGNORECASE)

    def url_rep(m: re.Match) -> str:
        host = _join_spoken(m.group(1))
        path = "".join("/" + _join_spoken(p) for p in re.split(r"\s*\bslash\s+", m.group(2).strip())[1:])
        return host + path

    # Email first: local part + "at" + host. Local part = last word before "at",
    # plus any immediately preceding single letters ("g suraneni", "j patel")
    # or dot/dash/underscore joins.
    email = re.compile(
        rf"\b((?:[A-Za-z]\s+)*{_W}(?:\s+(?:dot|dash|underscore)\s+{_W})*)\s+at\s+"
        rf"({_W}(?:\s+dot\s+{_W})*\s+dot\s+{TLDS})\b(?!\s+slash\b)", re.IGNORECASE)

    def email_rep(m: re.Match) -> str:
        local_words = m.group(1).split()
        # drop leading words that are clearly not part of an address ("me", "email")
        while len(local_words) > 1 and local_words[0].lower() in _STOP_LOCAL:
            local_words = local_words[1:]
        if local_words[-1].lower() in _STOP_LOCAL:
            return m.group(0)
        return f"{_join_spoken(' '.join(local_words))}@{_join_spoken(m.group(2))}"

    text = email.sub(email_rep, text)
    text = url.sub(url_rep, text)
    return text


def collapse_repeats(text: str) -> str:
    def rep(m: re.Match) -> str:
        return m.group(0) if m.group(1).lower() in _ALLOWED_REPEATS else m.group(1)
    return re.sub(r"\b(\w+)(?:,?\s+\1\b)+", rep, text, flags=re.IGNORECASE)


def split_breaks(text: str) -> list[tuple[str, str]]:
    """[(segment_text, break_after)], break_after in {"", "\\n", "\\n\\n"}."""
    parts = _BREAK.split(text)
    out: list[tuple[str, str]] = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip(" ,")
        brk = ""
        if i + 1 < len(parts):
            brk = "\n\n" if "paragraph" in parts[i + 1].lower() else "\n"
        if seg:
            out.append((seg, brk))
        elif out and brk:
            out[-1] = (out[-1][0], brk)
    return out or [("", "")]


def apply(text: str, cfg: dict | None = None) -> list[tuple[str, str]]:
    cfg = {"spoken_punctuation": True, "spoken_addresses": True, "repeated_words": True, **(cfg or {})}
    if cfg["spoken_addresses"]:
        text = spoken_addresses(text)
    if cfg["repeated_words"]:
        text = collapse_repeats(text)
    segments = split_breaks(text) if cfg["spoken_punctuation"] else [(text, "")]
    if cfg["spoken_punctuation"]:
        segments = [(spoken_punctuation(s), b) for s, b in segments]
    return segments


def join(segments: list[tuple[str, str]]) -> str:
    return "".join(s + b for s, b in segments).strip()
