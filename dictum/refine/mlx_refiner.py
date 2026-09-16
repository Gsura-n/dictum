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
from .common import refine_with
from .ollama_refiner_utils import build_messages


class MLXRefiner:
    name = "mlx"

    def __init__(self, model="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path=None,
                 max_tokens=300, dictionary=None, normalize=None, adapter_fallback=None,
                 use_personal=True, **_):
        self.model_id, self.max_tokens = model, max_tokens
        # Local adapter if it exists, else the published one, else the base model.
        from ..adapters import resolve
        self.adapter_path = resolve(adapter_path, use_personal) or resolve(adapter_fallback, False)
        self.entries = dict_mod.parse(dictionary)
        self.normalize_cfg = normalize
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
        return refine_with(lambda m, t: self._generate(m, t, self.max_tokens), self.name, transcript, mode,
                           self.entries, self.normalize_cfg)
