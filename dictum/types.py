"""Shared data types passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioClip:
    samples: np.ndarray        # float32, mono, shape (n,)
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples ** 2))) if len(self.samples) else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if len(self.samples) else 0.0

    @property
    def is_silent(self) -> bool:
        """Below roughly -60 dBFS: nothing was captured, or the wrong device is selected."""
        return self.rms < 1e-3


@dataclass
class Transcript:
    text: str
    engine: str
    latency_s: float           # wall-clock time inside the STT engine
    audio_s: float             # duration of the audio transcribed
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio time. Lower is better."""
        return self.latency_s / self.audio_s if self.audio_s else float("nan")


@dataclass
class Refined:
    text: str
    backend: str
    mode: str
    latency_s: float
    source: Transcript
    notes: list[str] = field(default_factory=list)   # guard trips, dictionary fixes
