from __future__ import annotations

from ..config import Mode
from ..types import Refined, Transcript


class Passthrough:
    """No LLM. Returns the transcript unchanged. Useful for benchmarking STT alone."""

    name = "passthrough"

    def warm_up(self, mode: Mode) -> float:
        return 0.0

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        return Refined(text=transcript.text, backend=self.name, mode=mode.name,
                       latency_s=0.0, source=transcript)
