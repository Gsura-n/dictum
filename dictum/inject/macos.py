"""Phase 3: type into the focused app. Needs Accessibility permission.

Strategy: put text on the clipboard, send Cmd+V, restore the old clipboard.
Simulated keystrokes per character are slower and mangle non-ASCII.
"""
from __future__ import annotations


class MacOSInjector:
    name = "macos"

    def inject(self, text: str) -> None:
        raise NotImplementedError("Phase 3: implement with Quartz CGEvent (Cmd+V) + pbpaste restore")
