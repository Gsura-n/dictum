from __future__ import annotations

from typing import Protocol

from ..config import Mode
from ..types import Refined, Transcript


class Refiner(Protocol):
    """Turns a raw transcript into text fit for the target (per mode)."""

    name: str

    def refine(self, transcript: Transcript, mode: Mode) -> Refined: ...
