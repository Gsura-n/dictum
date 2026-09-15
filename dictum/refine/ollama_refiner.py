"""LLM cleanup through a local Ollama server.

Design notes:
- Each mode carries its own system prompt, few-shot examples and optionally
  its own model. Small models follow examples far better than rules alone.
- keep_alive keeps weights resident so latency after the first call is the
  generation cost only. warm_up() pays the load cost at startup.
- Output is post-processed defensively: strip <think> blocks, code fences and
  wrapping quotes, since small models sometimes add them despite instructions.
"""
from __future__ import annotations

import re
import time

from ..config import Mode
from ..types import Refined, Transcript

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)


def clean_output(text: str) -> str:
    text = _THINK.sub("", text)
    text = _FENCE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def build_messages(mode: Mode, user_text: str, dictionary: list[str]) -> list[dict]:
    system = mode.prompt.strip()
    if dictionary:
        system += "\n\nSpell these names and terms exactly as written: " + ", ".join(dictionary) + "."
    msgs = [{"role": "system", "content": system}]
    for ex in mode.examples:
        msgs.append({"role": "user", "content": ex["in"]})
        msgs.append({"role": "assistant", "content": ex["out"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


class OllamaRefiner:
    name = "ollama"

    def __init__(self, host="http://localhost:11434", model="llama3.2:3b", timeout_s=15,
                 keep_alive="30m", num_predict=400, dictionary: list[str] | None = None, **_):
        import httpx  # lazy: optional dep
        self._client = httpx.Client(base_url=host, timeout=timeout_s)
        self.default_model = model
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.dictionary = dictionary or []

    def model_for(self, mode: Mode) -> str:
        return mode.model or self.default_model

    def warm_up(self, mode: Mode) -> float:
        """Load the mode's model into memory. Returns seconds taken."""
        t0 = time.perf_counter()
        r = self._client.post("/api/generate", json={"model": self.model_for(mode), "keep_alive": self.keep_alive})
        r.raise_for_status()
        return time.perf_counter() - t0

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        t0 = time.perf_counter()
        r = self._client.post("/api/chat", json={
            "model": self.model_for(mode),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1, "num_predict": self.num_predict},
            "messages": build_messages(mode, transcript.text, self.dictionary),
        })
        r.raise_for_status()
        text = clean_output(r.json()["message"]["content"])
        return Refined(text=text, backend=self.name, mode=mode.name,
                       latency_s=time.perf_counter() - t0, source=transcript)
