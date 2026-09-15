"""Refiner running a (optionally LoRA-adapted) model in-process with MLX.

Used for the fine-tuned dictation model. In-process generation avoids the HTTP
hop to Ollama, and MLX runs on the Apple Silicon GPU directly. MLX state is per
thread, so create and use this on the same thread (the app's processor thread).
"""
from __future__ import annotations

import time

from .. import dictionary as dict_mod
from ..config import Mode
from ..types import Refined, Transcript
from . import guards
from .ollama_refiner import build_messages, clean_output


class MLXRefiner:
    name = "mlx"

    def __init__(self, model="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path=None,
                 max_tokens=300, dictionary=None, **_):
        self.model_id, self.adapter_path, self.max_tokens = model, adapter_path, max_tokens
        self.entries = dict_mod.parse(dictionary)
        self._model = self._tok = None

    def model_for(self, mode: Mode) -> str:
        return f"{self.model_id}+{self.adapter_path}" if self.adapter_path else self.model_id

    def _load(self) -> None:
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_id, adapter_path=self.adapter_path)

    def _generate(self, mode: Mode, text: str, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        self._load()
        msgs = build_messages(mode, text, dict_mod.words(self.entries))
        prompt = self._tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(self._model, self._tok, prompt=prompt, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)

    def warm_up(self, mode: Mode) -> float:
        t0 = time.perf_counter()
        self._generate(mode, "test", 1)
        return time.perf_counter() - t0

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        t0 = time.perf_counter()
        source = dict_mod.apply(transcript.text, self.entries)
        raw_out = clean_output(self._generate(mode, source, self.max_tokens))
        text, reason = guards.check(mode.guard, source, raw_out)
        notes = [f"guard: {reason}; used transcript"] if reason else []
        if source != transcript.text:
            notes.append("dictionary applied")
        return Refined(text=text, backend=self.name, mode=mode.name,
                       latency_s=time.perf_counter() - t0, source=transcript, notes=notes)
