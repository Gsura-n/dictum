"""Keep pasted text from executing in a terminal.

Dictum only places text at the cursor; the user always presses Enter. A pasted
newline can run a command in shells without bracketed paste, so whenever the
frontmost app is a terminal (or unknown) the text is flattened to one line.
"""
from __future__ import annotations

import unicodedata

from .context import AppInfo

NOTES = {
    "terminal": "terminal in front; flattened to one line",
    "unknown": "frontmost app unknown; flattened for safety",
}


def terminal_safe(text: str) -> str:
    """One line, single spaces between lines, no control characters, no trailing whitespace."""
    parts = []
    for line in text.splitlines():           # \n \r\n \r \v \f \x1c-\x1e \x85    
        line = line.replace("\t", " ")
        line = "".join(c for c in line if unicodedata.category(c) != "Cc")
        if line.strip():
            parts.append(line)
    if not parts:
        return ""
    # Single-line text keeps its leading spaces; joined lines are trimmed at the seams.
    return " ".join([parts[0].rstrip(), *(p.strip() for p in parts[1:])])


def flatten_reason(apps_cfg: dict, *apps: AppInfo | None, fail_closed: bool = True) -> str | None:
    """Why the text must be flattened ("terminal", "unknown"), or None if it is safe as is.

    Any of the given apps being a terminal wins; an unknown app (None) counts as unsafe
    when fail_closed. Terminals are the ids in apps.terminals, matched like apps.rules.
    """
    terminals = {t.lower() for t in (apps_cfg or {}).get("terminals", []) or []}
    if any(a is not None and (a.bundle_id.lower() in terminals or a.name.lower() in terminals) for a in apps):
        return "terminal"
    if fail_closed and any(a is None for a in apps):
        return "unknown"
    return None
