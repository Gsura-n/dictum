"""Wires the four stages together. This is the only place they meet."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .capture import AudioCapture, SoundDeviceCapture
from .config import Config
from .inject import Injector, create_injector
from .refine import Refiner, create_refiner
from .stt import STTEngine, create_engine
from .types import AudioClip, Refined


@dataclass
class Timing:
    audio_s: float
    stt_s: float
    refine_s: float
    total_s: float


class Pipeline:
    def __init__(self, capture: AudioCapture, stt: STTEngine, refiner: Refiner, injector: Injector, cfg: Config):
        self.capture, self.stt, self.refiner, self.injector, self.cfg = capture, stt, refiner, injector, cfg

    @classmethod
    def from_config(cls, cfg: Config, engine: str | None = None) -> "Pipeline":
        a = cfg.audio
        capture = SoundDeviceCapture(a.get("sample_rate", 16000), a.get("channels", 1), a.get("device"))
        name, ecfg = cfg.stt_engine_config(engine)
        stt = create_engine(name, **ecfg)
        refiner = create_refiner(cfg.refine, cfg.dictionary)
        injector = create_injector(cfg.inject)
        return cls(capture, stt, refiner, injector, cfg)

    def warm_up(self) -> float:
        t0 = time.perf_counter()
        self.stt.load()
        return time.perf_counter() - t0

    def process(self, clip: AudioClip, mode_name: str | None = None) -> tuple[Refined, Timing]:
        """Everything after the mic: STT -> refine -> inject."""
        t0 = time.perf_counter()
        transcript = self.stt.transcribe(clip)
        refined = self.refiner.refine(transcript, self.cfg.mode(mode_name))
        self.injector.inject(refined.text)
        total = time.perf_counter() - t0
        return refined, Timing(clip.duration_s, transcript.latency_s, refined.latency_s, total)
