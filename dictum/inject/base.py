from __future__ import annotations

from typing import Protocol


class Injector(Protocol):
    """Delivers final text to its destination (terminal, clipboard, focused app)."""

    name: str

    def inject(self, text: str) -> None: ...
