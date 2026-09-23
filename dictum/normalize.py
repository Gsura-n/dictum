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


_PREP = {"in", "on", "at", "by", "for", "from", "to", "with", "of", "into", "onto",
         "after", "before", "during", "until", "since", "about", "over", "under"}
_DET = {"the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "his", "her", "their"}
_WORD = re.compile(r"[A-Za-z0-9']+")
_HARD_STOP = re.compile(r"[.?!;]")


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(w == h for h in it) for w in needle)


def collapse_restarts(text: str, max_span: int = 6, slack: int = 4) -> str:
    """Drop an abandoned false start when the speaker starts the phrase again.

    "how would be how I would be able" -> "how I would be able"
    "I went to the store I went to the market" -> "I went to the market"

    A span is only dropped when the speaker clearly restarted it: the span and
    the words that follow begin with the same word, and every word of the span
    reappears, in order, in the words that follow. That last test is what keeps
    ordinary English intact - "the cat sat on the mat" repeats "the" but does
    not repeat "cat sat on", so nothing is dropped.
    """
    for _ in range(10):                      # a long take can hold several
        toks = _tokens(text)
        words = [t[0].lower() for t in toks]
        cut, blocked = None, []
        for i in range(len(words)):
            if any(a <= i < b for a, b in blocked):
                continue                     # inside a span we already cleared
            for j in range(i + 2, min(i + max_span + 1, len(words))):
                if words[i] != words[j]:
                    continue
                span = words[i:j]
                if _HARD_STOP.search(text[toks[i][1]:toks[j][1]]):
                    break                    # never reach across a sentence end
                window = words[j:j + len(span) + slack]
                if len(window) < len(span) or not _is_subsequence(span, window):
                    continue
                if words[j:j + len(span)] == span and i and words[i - 1] in _PREP and span[0] in _DET:
                    blocked.append((i, j))   # "in the end | the end justifies" is English
                    break
                cut = (toks[i][1], toks[j][1])
                break
            if cut:
                break
        if not cut:
            return text
        text = (text[:cut[0]] + text[cut[1]:]).strip()
    return text


# "better and better and better and better" -> "better and better".
# Three or more copies is emphasis that reads as a stutter in writing; two keeps
# the idiom. One to three words per copy so "salt and pepper and salt and
# pepper and salt and pepper" also settles down.
_PHRASE_RUN = re.compile(r"\b(\w+(?:\s+\w+){0,2})((?:\s+(and|or)\s+\1\b){2,})", re.IGNORECASE)


def reduce_phrase_repeats(text: str) -> str:
    def rep(m: re.Match) -> str:
        return f"{m.group(1)} {m.group(3)} {m.group(1)}"
    return _PHRASE_RUN.sub(rep, text)


# "scratch that" / "strike that" as an editing command. Not when it is a real
# verb phrase: "don't scratch that", "scratch that itch", "strike that off the list".
_SCRATCH = re.compile(
    r"(?<!\bto )(?<!\bnot )(?<!n't )\b(?:scratch|strike) that\b(?!\s+(?:itch|off|out|from|one|line|word|part))[\s,.!;:-]*",
    re.IGNORECASE)
_BOUNDARY = re.compile(r"[.?!;](?=\s|$)")


def scratch_that(text: str) -> str:
    """Delete what the speaker retracted.

    "X Y scratch that Z"            -> "Z"   (retracts the sentence it is in)
    "X Y. Scratch that. Z"          -> "Z"   (starts a sentence: retracts the previous one)
    Never reaches past a paragraph break; call it per segment.
    """
    for _ in range(10):                      # repeated commands in one segment
        m = _SCRATCH.search(text)
        if not m:
            break
        before, after = text[:m.start()], text[m.end():]
        bounds = [b.end() for b in _BOUNDARY.finditer(before)]
        cut = bounds[-1] if bounds else 0
        if not before[cut:].strip():         # command starts its own sentence: retract the previous sentence
            cut = bounds[-2] if len(bounds) >= 2 else 0
        keep = before[:cut].rstrip()
        after = after.lstrip()
        if after and (not keep or keep[-1] in ".?!"):
            after = after[0].upper() + after[1:]
        text = (keep + " " + after).strip()
    return text


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
    cfg = {"spoken_punctuation": True, "spoken_addresses": True, "repeated_words": True,
           "scratch_that": True, "restarts": True, "phrase_repeats": True, **(cfg or {})}
    if cfg["spoken_addresses"]:
        text = spoken_addresses(text)
    if cfg["repeated_words"]:
        text = collapse_repeats(text)
    if cfg["restarts"]:
        text = collapse_restarts(text)
    if cfg["phrase_repeats"]:
        text = reduce_phrase_repeats(text)
    segments = split_breaks(text) if cfg["spoken_punctuation"] else [(text, "")]
    if cfg["spoken_punctuation"]:
        segments = [(spoken_punctuation(s), b) for s, b in segments]
    if cfg["scratch_that"]:
        segments = [(scratch_that(s), b) for s, b in segments]
        segments = [(s, b) for s, b in segments if s.strip()] or [("", "")]
    return segments


def join(segments: list[tuple[str, str]]) -> str:
    return "".join(s + b for s, b in segments).strip()


_SENT = re.compile(r"(?<=[.?!])\s+")
_CORRECTION_START = re.compile(r"^(no|not|nope|actually|sorry|wait|i mean|i meant|or rather|rather|make that|"
                               r"scratch that|strike that|oops|instead)\b", re.IGNORECASE)


_CLAUSE_START = re.compile(
    r"\s+(?=(?:and so|and also|and then|and of course|but|so|because|also|of course|let's say|"
    r"i'm also|i am also|and i'm|and i|and the|and when|and if|then|which|anyway)\b)",
    re.IGNORECASE)


def _split_run_on(sentence: str, max_words: int) -> list[str]:
    """Split one over-long sentence at clause starters ("and I'm also", "so",
    "because"), aiming for pieces between half and the full max size. Falls
    back to a hard split at max_words if no clause starter is found."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    pieces, rest = [], sentence
    while len(rest.split()) > int(max_words * 1.3):
        cut = None
        for m in _CLAUSE_START.finditer(rest):
            n_before = len(rest[:m.start()].split())
            if max_words // 2 <= n_before <= max_words:
                cut = m.start()                  # last clause start within the target size
            elif cut is None and max_words < n_before <= int(max_words * 1.3):
                cut = m.start()                  # slight overshoot beats splitting mid-phrase
                break
        if cut is None:
            w = rest.split()
            cut = len(" ".join(w[:max_words]))
        else:
            # never strand the start of a clause: "... and I'm | also" -> "... | and I'm also"
            while True:
                head = rest[:cut].rstrip()
                last = head.rsplit(" ", 1)[-1].lower().strip(",")
                if last in {"and", "i'm", "i", "am", "we", "we're", "so"} and len(head.split()) > max_words // 2:
                    cut = len(head) - len(head.rsplit(" ", 1)[-1])
                else:
                    break
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def chunk(text: str, max_words: int) -> list[str]:
    """Split long text into chunks of about max_words at sentence boundaries.

    A sentence that starts with a correction ("No, not Monday.") is never
    separated from the sentence before it, since the correction refers back to it.
    """
    if max_words <= 0 or len(text.split()) <= max_words:
        return [text]
    sentences = [x for x in _SENT.split(text.strip()) if x]
    # Long unpunctuated speech (the STT often emits none for a continuous
    # monologue): break run-on "sentences" at clause starters instead.
    sentences = [piece for sent in sentences for piece in _split_run_on(sent, max_words)]
    chunks: list[list[str]] = []
    for sent in sentences:
        n_cur = sum(len(x.split()) for x in chunks[-1]) if chunks else 0
        glue = bool(chunks) and bool(_CORRECTION_START.match(sent))
        if chunks and (glue or n_cur + len(sent.split()) <= max_words):
            chunks[-1].append(sent)
        else:
            chunks.append([sent])
    # a correction glued onto a full chunk may pull the next sentence too; fine
    return [" ".join(c) for c in chunks]
