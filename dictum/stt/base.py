from __future__ import annotations

from typing import Protocol

from ..types import AudioClip, Transcript


class STTEngine(Protocol):
    """Speech-to-text adapter. Implementations wrap one model family."""

    name: str

    def load(self) -> None:
        """Load weights into memory. Called once; may download on first use."""
        ...

    def transcribe(self, clip: AudioClip) -> Transcript: ...
