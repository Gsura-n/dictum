"""Prompt lookup decoding: greedy generation that drafts tokens by copying from the context.

Cleanup output mostly repeats the input. After n tokens are generated, find the same n tokens earlier in the
context (prompt plus output so far) and propose the k tokens that followed them. One forward over the last
token plus the draft checks all of it; the longest prefix equal to the model's own argmax is kept, the rest is
trimmed from the cache. The output is the same as one-token-at-a-time greedy decoding.

No MLX import here: the model and the cache trim are passed in, so the loop is testable on any OS.
  forward(tokens, n_out) feeds tokens into the cache and returns the argmax after each of the last n_out of them
  trim(n) drops the last n tokens from the cache
The cache is left holding whatever was fed and not trimmed; the caller restores it (see prefix_cache.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class LookupStats:
    steps: int = 0          # verify forwards
    drafted: int = 0        # tokens proposed
    accepted: int = 0       # tokens proposed and kept


def draft(ctx: list[int], ngram: int, k: int) -> list[int]:
    """Up to k tokens that followed the latest earlier occurrence of the last ngram tokens of ctx. O(len(ctx))."""
    if len(ctx) <= ngram:
        return []
    key = ctx[-ngram:]
    for i in range(len(ctx) - ngram - 1, -1, -1):
        if ctx[i:i + ngram] == key:
            return ctx[i + ngram:i + ngram + k]
    return []


def lookup_generate(forward: Callable[[list[int], int], list[int]], trim: Callable[[int], None], ids: list[int],
                    *, max_tokens: int, eos: set[int], ngram: int = 2, draft_tokens: int = 4,
                    stats: LookupStats | None = None) -> list[int]:
    """Token ids generated from ids, without the EOS. Stops on EOS or after max_tokens tokens, like stream_generate."""
    ctx, out = list(ids), []
    last = forward(ctx, 1)[0]
    while True:
        out.append(last)
        ctx.append(last)
        if last in eos or len(out) >= max_tokens:
            break
        d = draft(ctx, ngram, min(draft_tokens, max_tokens - len(out)))
        preds = forward([last] + d, len(d) + 1)
        a = 0
        while a < len(d) and preds[a] == d[a]:
            a += 1
        if stats is not None:
            stats.steps += 1
            stats.drafted += len(d)
            stats.accepted += a
        if a < len(d):
            trim(len(d) - a)
        done = False
        for t in d[:a]:
            out.append(t)
            ctx.append(t)
            if t in eos or len(out) >= max_tokens:
                done = True
                break
        if done:
            break
        last = preds[a]
    return out[:-1] if out[-1] in eos else out
