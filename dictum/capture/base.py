from __future__ import annotations

from typing import Protocol

from ..types import AudioClip


class AudioCapture(Protocol):
    """Records microphone audio between start() and stop()."""

    def start(self) -> None: ...
    def stop(self) -> AudioClip: ...
