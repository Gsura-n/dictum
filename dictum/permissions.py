"""macOS privacy permission checks. Both are granted per terminal app."""
from __future__ import annotations

import sys


def _q(name):
    try:
        import Quartz
        return getattr(Quartz, name, None)
    except ImportError:
        return None


def input_monitoring(request: bool = False) -> bool | None:
    """Needed to see global key presses (the hotkey)."""
    if sys.platform != "darwin":
        return None
    fn = _q("CGRequestListenEventAccess" if request else "CGPreflightListenEventAccess")
    return bool(fn()) if fn else None


def accessibility(request: bool = False) -> bool | None:
    """Needed to post the Cmd+V keystroke into other apps."""
    if sys.platform != "darwin":
        return None
    fn = _q("CGRequestPostEventAccess" if request else "CGPreflightPostEventAccess")
    return bool(fn()) if fn else None
