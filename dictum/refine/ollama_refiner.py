"""Phase 2: LLM cleanup through a local Ollama server."""
from __future__ import annotations

import time

from ..config import Mode
from ..types import Refined, Transcript


class OllamaRefiner:
    name = "ollama"

    def __init__(self, host="http://localhost:11434", model="qwen2.5:3b", timeout_s=10,
                 dictionary: list[str] | None = None, **_):
        import httpx  # lazy: optional dep
        self._client = httpx.Client(base_url=host, timeout=timeout_s)
        self.model = model
        self.dictionary = dictionary or []

    def _system(self, mode: Mode) -> str:
        s = mode.prompt.strip()
        if self.dictionary:
            s += "\n\nSpell these names and terms exactly: " + ", ".join(self.dictionary) + "."
        return s

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        t0 = time.perf_counter()
        r = self._client.post("/api/chat", json={
            "model": self.model, "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": self._system(mode)},
                {"role": "user", "content": transcript.text},
            ],
        })
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        return Refined(text=text, backend=self.name, mode=mode.name,
                       latency_s=time.perf_counter() - t0, source=transcript)
