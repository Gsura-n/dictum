"""NVIDIA Parakeet TDT via the parakeet-mlx port (Apple Silicon)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import soundfile as sf

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

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.load()
        # parakeet-mlx reads from a file path; write the clip to a temp wav.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, clip.samples, clip.sample_rate)
            t0 = time.perf_counter()
            result = self._model.transcribe(Path(tmp.name))
            dt = time.perf_counter() - t0
        return Transcript(
            text=result.text.strip(), engine=self.name, latency_s=dt,
            audio_s=clip.duration_s,
            extra={"model": self.model_id, "sentences": len(getattr(result, "sentences", []) or [])},
        )
