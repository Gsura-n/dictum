from __future__ import annotations

from ..config import Mode
from ..types import Refined, Transcript


class Passthrough:
    """Phase 1: no LLM. Returns the transcript unchanged."""

    name = "passthrough"

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        return Refined(text=transcript.text, backend=self.name, mode=mode.name,
                       latency_s=0.0, source=transcript)
