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
from .prefix_cache import PrefixCache, split_prompt


class MLXRefiner:
    name = "mlx"

    def __init__(self, model="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path=None,
                 max_tokens=300, dictionary=None, normalize=None, adapter_fallback=None,
                 use_personal=True, prefix_cache=True, **_):
        self.model_id, self.max_tokens = model, max_tokens
        # Local adapter if it exists, else the published one, else the base model.
        from ..adapters import resolve
        self.adapter_path = resolve(adapter_path, use_personal) or resolve(adapter_fallback, False)
        self.entries = dict_mod.parse(dictionary)
        self.normalize_cfg = normalize
        self._model = self._tok = None
        self.prefix_cache = PrefixCache(make_cache=self._make_cache, prefill=self._prefill_prefix,
                                        offset=lambda c: c[0].offset, trim=self._trim_cache,
                                        can_trim=self._can_trim, enabled=prefix_cache)

    cache_hits = property(lambda self: self.prefix_cache.hits)
    cache_misses = property(lambda self: self.prefix_cache.misses)

    def model_for(self, mode: Mode) -> str:
        return f"{self.model_id}+{self.adapter_path}" if self.adapter_path else self.model_id

    def _load(self) -> None:
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_id, adapter_path=self.adapter_path)

    def _make_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self._model)

    def _prefill_prefix(self, cache, ids) -> None:
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        with mx.stream(generation_stream):
            self._model(mx.array(ids)[None], cache=cache)
            mx.eval([c.state for c in cache])

    @staticmethod
    def _can_trim(cache) -> bool:
        from mlx_lm.models.cache import can_trim_prompt_cache
        return can_trim_prompt_cache(cache)

    @staticmethod
    def _trim_cache(cache, n: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache
        if trim_prompt_cache(cache, n) != n:
            raise RuntimeError("cache did not trim the requested number of tokens")

    def _encode(self, text: str) -> list[int]:
        # same choice mlx_lm.stream_generate makes for a string prompt
        bos = self._tok.bos_token
        return self._tok.encode(text, add_special_tokens=bos is None or not text.startswith(bos))

    def _generate(self, mode: Mode, text: str, max_tokens: int) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self._load()
        msgs = build_messages(mode, text, dict_mod.words(self.entries))

        def render(m, add_generation_prompt):
            return self._tok.apply_chat_template(m, add_generation_prompt=add_generation_prompt, tokenize=False)

        parts = split_prompt(render, msgs)
        prefix, full = parts if parts else (None, render(msgs, True))
        sampler = make_sampler(temp=0.0)

        def gen(ids, cache):
            extra = {} if cache is None else {"prompt_cache": cache}
            return "".join(r.text for r in stream_generate(self._model, self._tok, prompt=ids,
                                                           max_tokens=max_tokens, sampler=sampler, **extra))

        return self.prefix_cache.run(mode.name, (self.model_id, self.adapter_path), prefix, full, self._encode, gen)

    def warm_up(self, mode: Mode) -> float:
        t0 = time.perf_counter()
        self._generate(mode, "test", 1)
        return time.perf_counter() - t0

    def refine(self, transcript: Transcript, mode: Mode) -> Refined:
        return refine_with(lambda m, t: self._generate(m, t, self.max_tokens), self.name, transcript, mode,
                           self.entries, self.normalize_cfg)
