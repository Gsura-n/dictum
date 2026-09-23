"""Wires the four stages together. This is the only place they meet."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .capture import AudioCapture, SoundDeviceCapture
from .config import Config
from .inject import Injector, create_injector
from .refine import Refiner, create_refiner
from .refine.common import refine_levels
from .stt import STTEngine, create_engine
from .types import AudioClip, Refined, Transcript


@dataclass
class Timing:
    audio_s: float
    stt_s: float
    refine_s: float
    total_s: float

    def __str__(self) -> str:
        rtf = self.stt_s / self.audio_s if self.audio_s else 0
        return (f"audio {self.audio_s:.1f}s | stt {self.stt_s:.2f}s (rtf {rtf:.2f}) "
                f"| refine {self.refine_s:.2f}s | total {self.total_s:.2f}s")


class Pipeline:
    def __init__(self, capture: AudioCapture, stt: STTEngine, refiner: Refiner, injector: Injector, cfg: Config):
        self.capture, self.stt, self.refiner, self.injector, self.cfg = capture, stt, refiner, injector, cfg

    @classmethod
    def from_config(cls, cfg: Config, engine: str | None = None, refine: str | None = None,
                    inject: str | None = None, device: int | None = None) -> "Pipeline":
        a = cfg.audio
        dev = device if device is not None else a.get("device")
        capture = SoundDeviceCapture(a.get("sample_rate", 16000), a.get("channels", 1), dev)
        name, ecfg = cfg.stt_engine_config(engine)
        stt = create_engine(name, **ecfg)
        refiner = create_refiner(cfg.refine, cfg.dictionary, backend=refine)
        icfg = dict(cfg.inject)
        if inject:
            icfg["backend"] = inject
        injector = create_injector(icfg)
        return cls(capture, stt, refiner, injector, cfg)

    def warm_up(self, mode_name: str | None = None, modes: list[str] | None = None) -> dict[str, float]:
        """Load STT weights and the refiner model(s) for the given mode(s)."""
        t0 = time.perf_counter()
        self.stt.load()
        stt_s = time.perf_counter() - t0
        refine_s = 0.0
        for m in (modes or [mode_name]):
            try:
                refine_s += self.refiner.warm_up(self.cfg.mode(m))
            except Exception as e:
                print(f"[warm-up] refiner for mode '{m}' unavailable: {e}", flush=True)
        return {"stt": stt_s, "refine": refine_s}

    def process(self, clip: AudioClip, mode_name: str | None = None) -> tuple[Refined, Timing]:
        """Everything after the mic: STT -> refine -> inject."""
        t0 = time.perf_counter()
        transcript = self.stt.transcribe(clip)
        if len(transcript.text.split()) < 2:
            # Nothing (or one word) to clean up. Never hand an LLM an empty
            # prompt: it will happily invent a sentence.
            refined = Refined(text=transcript.text, backend="skipped", mode=mode_name or "",
                              latency_s=0.0, source=transcript)
        else:
            try:
                refined = self.refine_text(transcript, mode_name)
            except Exception as e:  # Ollama down, model missing, timeout...
                # Degrade to the raw transcript rather than losing what was said.
                refined = Refined(text=transcript.text, backend=f"failed ({type(e).__name__}: {e})",
                                  mode=mode_name or "", latency_s=0.0, source=transcript)
        self.injector.inject(refined.text)
        if self.cfg.raw.get("history", {}).get("enabled") and refined.text.strip():
            try:
                from . import history
                refined.notes.append(f"history #{history.record(transcript.text, refined.text, refined.mode)}")
            except Exception as e:
                refined.notes.append(f"history not saved: {e}")
        total = time.perf_counter() - t0
        return refined, Timing(clip.duration_s, transcript.latency_s, refined.latency_s, total)

    def refine_text(self, transcript: Transcript, mode_name: str | None = None) -> Refined:
        mode = self.cfg.mode(mode_name)
        n = len(transcript.text.split())
        if mode.max_input_words and n > mode.max_input_words and mode.fallback_mode:
            # e.g. a long ramble in a terminal is dictation, not a shell command
            refined = self.refiner.refine(transcript, self.cfg.mode(mode.fallback_mode))
            refined.notes.insert(0, f"{n} words is too long for {mode.name}; used {mode.fallback_mode}")
            return refined
        if mode.pre_mode:
            return refine_levels(self.refiner, self.cfg, transcript, mode)
        refined = self.refiner.refine(transcript, mode)
        if mode.name.startswith("command") and mode.fallback_mode:
            from .refine.guards import looks_like_command
            if not looks_like_command(refined.text):
                # The model answered in words instead of producing a command, so this
                # was probably dictation. Paste cleaned text rather than a bogus command.
                fb = self.refiner.refine(transcript, self.cfg.mode(mode.fallback_mode))
                fb.notes.insert(0, f"'{refined.text[:40]}' is not a runnable command; used {mode.fallback_mode}")
                return fb
        return refined
