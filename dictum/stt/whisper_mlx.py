"""OpenAI Whisper via mlx-whisper (Apple Silicon)."""
from __future__ import annotations

import time

from ..types import AudioClip, Transcript


class WhisperMLX:
    name = "whisper"

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo", language: str | None = "en", **_):
        self.model_id = model
        self.language = language
        self._mod = None

    def load(self) -> None:
        if self._mod is None:
            import mlx_whisper  # lazy: optional dep
            self._mod = mlx_whisper
            # Warm up so the first real call doesn't pay the weight-load cost.
            import numpy as np
            self._mod.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.model_id)

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.load()
        assert clip.sample_rate == 16000, "whisper expects 16 kHz audio"
        t0 = time.perf_counter()
        result = self._mod.transcribe(
            clip.samples, path_or_hf_repo=self.model_id, language=self.language, fp16=True,
        )
        dt = time.perf_counter() - t0
        return Transcript(
            text=result["text"].strip(), engine=self.name, latency_s=dt,
            audio_s=clip.duration_s, extra={"model": self.model_id},
        )
