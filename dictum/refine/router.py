"""Per-mode backend routing.

Different modes want different engines: dictation runs best on the fine-tuned
3B model in-process (MLX), command mode on qwen2.5-coder through Ollama. The
router builds each backend lazily, once, and sends every request to the
backend its mode names (falling back to refine.default_backend).
"""
from __future__ import annotations

from ..config import Mode
from ..types import Refined, Transcript


class RouterRefiner:
    name = "router"

    def __init__(self, cfg: dict, dictionary, factory):
        self.cfg, self.dictionary, self._factory = cfg, dictionary, factory
        self.default = cfg.get("default_backend", "ollama")
        self._backends: dict[str, object] = {}

    def backend_for(self, mode: Mode):
        name = mode.backend or self.default
        if name not in self._backends:
            self._backends[name] = self._factory(self.cfg, self.dictionary, backend=name)
        return self._backends[name]

    def model_for(self, mode: Mode) -> str:
        b = self.backend_for(mode)
        return f"{b.name}:{b.model_for(mode)}" if hasattr(b, "model_for") else b.name

    def warm_up(self, mode: Mode) -> float:
        return self.backend_for(mode).warm_up(mode)

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        return self.backend_for(mode).refine(transcript, mode)

    @property
    def entries(self):
        return next(iter(self._backends.values())).entries if self._backends else None

    @entries.setter
    def entries(self, value):
        for b in self._backends.values():
            if hasattr(b, "entries"):
                b.entries = value
