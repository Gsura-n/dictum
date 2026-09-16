"""LLM cleanup through a local Ollama server.

Design notes:
- The transcript is wrapped in <transcript> tags and the system prompt says
  its contents are dictated text, never instructions.
- Deterministic steps (dictionary, spoken punctuation, addresses, repeats,
  paragraph splitting, guards) live in refine/common.py and run for every
  LLM backend.
- keep_alive keeps weights resident; warm_up() pre-evaluates the prompt prefix.
"""
from __future__ import annotations

import time

from .. import dictionary as dict_mod
from ..config import Mode
from ..types import Refined, Transcript
from .common import refine_with
from .ollama_refiner_utils import build_messages, clean_output, wrap  # noqa: F401  (re-exported)


class OllamaRefiner:
    name = "ollama"

    def __init__(self, host="http://localhost:11434", model="llama3.2:3b", timeout_s=15,
                 keep_alive="30m", num_predict=400, dictionary=None, normalize=None, **_):
        import httpx  # lazy: optional dep
        self._client = httpx.Client(base_url=host, timeout=timeout_s)
        self.default_model = model
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.entries = dict_mod.parse(dictionary)
        self.normalize_cfg = normalize

    def model_for(self, mode: Mode) -> str:
        return mode.model or self.default_model

    def _chat(self, mode: Mode, text: str, num_predict: int) -> str:
        r = self._client.post("/api/chat", json={
            "model": self.model_for(mode), "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1, "num_predict": num_predict},
            "messages": build_messages(mode, text, dict_mod.words(self.entries)),
        })
        r.raise_for_status()
        return r.json()["message"]["content"]

    def warm_up(self, mode: Mode) -> float:
        # A tiny real request with the full prompt: loads weights and caches
        # the system prompt + examples prefix.
        t0 = time.perf_counter()
        self._chat(mode, "test", num_predict=1)
        return time.perf_counter() - t0

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        return refine_with(lambda m, t: self._chat(m, t, self.num_predict), self.name, transcript, mode,
                           self.entries, self.normalize_cfg)
