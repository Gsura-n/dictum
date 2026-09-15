"""NVIDIA Parakeet TDT via the parakeet-mlx port (Apple Silicon).

We bypass model.transcribe(path), which shells out to ffmpeg to decode a file.
The audio is already in memory, so we compute the log-mel ourselves and call
model.generate() directly. No temp file, no ffmpeg, one less process spawn.
"""
from __future__ import annotations

import time

import numpy as np

from ..types import AudioClip, Transcript


class ParakeetMLX:
    name = "parakeet"

    def __init__(self, model: str = "mlx-community/parakeet-tdt-0.6b-v2", **_):
        self.model_id = model
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from parakeet_mlx import from_pretrained  # lazy: optional dep
            self._model = from_pretrained(self.model_id)
            # Warm up the graph so the first real utterance is not slow.
            self._transcribe_samples(np.zeros(16000, dtype=np.float32), 16000)

    def _transcribe_samples(self, samples: np.ndarray, sample_rate: int):
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        cfg = self._model.preprocessor_config
        assert sample_rate == cfg.sample_rate, f"parakeet expects {cfg.sample_rate} Hz audio"
        audio = mx.array(samples.astype(np.float32))  # float32: get_logmel relies on it
        mel = get_logmel(audio, cfg)
        return self._model.generate(mel)[0]

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.load()
        t0 = time.perf_counter()
        result = self._transcribe_samples(clip.samples, clip.sample_rate)
        dt = time.perf_counter() - t0
        return Transcript(
            text=result.text.strip(), engine=self.name, latency_s=dt,
            audio_s=clip.duration_s,
            extra={"model": self.model_id, "sentences": len(getattr(result, "sentences", []) or [])},
        )
