"""Reuse the fixed part of a prompt (system prompt plus few-shot examples) between calls.

The MLX refiner prefills the ~70 fixed tokens once, keeps the KV cache, and after each call trims it back to
the prefix, so every utterance starts from the same state. No MLX import here: the cache operations are passed
in, so the logic is testable on any OS.

A call that does not use a warm cache is a miss. Each miss has a reason, is counted and is logged once at debug
level: disabled, no_prefix_match, token_mismatch, non_trimmable, first_use, prefix_changed, thread_rebuild,
trim_failed. MLX state belongs to the thread that made it, so a cache built on another thread is discarded and
rebuilt on the calling thread.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


def split_prompt(render: Callable[[list[dict], bool], str], messages: list[dict]) -> tuple[str, str] | None:
    """(prefix_text, full_text), the prefix being everything before the last user turn. None if the full prompt
    does not start with it (a template that stamps the time, with midnight between the renders)."""
    if len(messages) < 2:
        return None
    full = render(messages, True)
    prefix = render(messages[:-1], False)
    return (prefix, full) if prefix and full.startswith(prefix) else None


@dataclass
class _Entry:
    ident: tuple          # what the cache was built for: (model, adapter) and the prefix tokens
    cache: Any            # None marks "this prefix cannot be cached"
    thread: int
    n_prefix: int


class PrefixCache:
    def __init__(self, *, make_cache: Callable[[], Any], prefill: Callable[[Any, list], None],
                 offset: Callable[[Any], int], trim: Callable[[Any, int], None],
                 can_trim: Callable[[Any], bool], enabled: bool = True):
        self._make, self._prefill, self._offset, self._trim, self._can_trim = make_cache, prefill, offset, trim, can_trim
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.miss_reasons: Counter = Counter()
        self._entries: dict[str, _Entry] = {}
        self._dropped: dict[str, str] = {}      # mode -> why its entry was dropped, reported by the next miss

    def _miss(self, reason: str, mode: str) -> None:
        self.misses += 1
        self.miss_reasons[reason] += 1
        log.debug("prefix cache miss: %s (mode %s)", reason, mode)

    def _plain(self, reason: str, mode: str, full_ids: list, gen: Callable) -> str:
        self._miss(reason, mode)
        return gen(full_ids, None)

    def run(self, mode: str, scope: tuple, prefix_text: str | None, full_text: str,
            encode: Callable[[str], list], gen: Callable[[list, Any], str]) -> str:
        """gen(ids, cache) generates from the tokens still to feed. cache is None for an ordinary run."""
        full_ids = encode(full_text)
        if not self.enabled:
            return self._plain("disabled", mode, full_ids, gen)
        if prefix_text is None:
            return self._plain("no_prefix_match", mode, full_ids, gen)
        prefix_ids = encode(prefix_text)
        n = len(prefix_ids)
        if full_ids[:n] != prefix_ids:           # tokenizing the pieces must give the tokens of the whole
            return self._plain("token_mismatch", mode, full_ids, gen)

        ident, me = (scope, tuple(prefix_ids)), threading.get_ident()
        entry, reason = self._entries.get(mode), self._dropped.pop(mode, "first_use")
        if entry is not None and entry.ident != ident:
            entry, reason = None, "prefix_changed"
        elif entry is not None and entry.cache is not None and entry.thread != me:
            entry, reason = None, "thread_rebuild"
        if entry is not None and entry.cache is None:
            return self._plain("non_trimmable", mode, full_ids, gen)
        if entry is None:
            cache = self._make()
            if not self._can_trim(cache):
                self._entries[mode] = _Entry(ident, None, me, 0)
                return self._plain("non_trimmable", mode, full_ids, gen)
            self._prefill(cache, prefix_ids)
            entry = self._entries[mode] = _Entry(ident, cache, me, n)
            self._miss(reason, mode)
        else:
            self.hits += 1
        try:
            return gen(full_ids[n:], entry.cache)
        finally:
            self._restore(mode, entry)

    def _restore(self, mode: str, entry: _Entry) -> None:
        """Trim back to the prefix. If that cannot be confirmed, drop the entry so nothing can leak."""
        try:
            extra = self._offset(entry.cache) - entry.n_prefix
            if extra:
                self._trim(entry.cache, extra)
            ok = self._offset(entry.cache) == entry.n_prefix
        except Exception:
            ok = False
        if not ok:
            self._entries.pop(mode, None)
            self._dropped[mode] = "trim_failed"
