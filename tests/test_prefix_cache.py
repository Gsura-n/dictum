"""Prefix cache for the MLX refiner, tested without MLX: fake template, tokenizer and cache. The fake model's
output is a function of everything it was fed, so a leak from one utterance into the next changes it."""
import logging
import re
import threading
from types import SimpleNamespace

import pytest

from dictum.refine.prefix_cache import PrefixCache, split_prompt

LOG = "dictum.refine.prefix_cache"


# ---- fakes -----------------------------------------------------------------------------------------------
def render(messages, add_generation_prompt):
    out = "<|bos|>" + "".join(f"<|{m['role']}|>{m['content']}<|eot|>" for m in messages)
    return out + ("<|assistant|>" if add_generation_prompt else "")


_TOK = re.compile(r"<\|\w+\|>|[A-Za-z]+|\s+|.")


def encode(text):
    """Special markers are one token; letters merge, so a cut in the middle of a word changes the tokens."""
    return _TOK.findall(text)


def msgs(user="hello there", system="clean it up", examples=()):
    m = [{"role": "system", "content": system}]
    for a, b in examples:
        m += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    return m + [{"role": "user", "content": user}]


class Harness:
    """A PrefixCache wired to fakes, counting what the real MLX calls would have done."""

    def __init__(self, enabled=True, trimmable=True, trim_works=True):
        self.prefills, self.made, self.trim_works, self.seen = 0, 0, trim_works, []   # seen: context per call

        def make_cache():
            self.made += 1
            return SimpleNamespace(tokens=[], trimmable=trimmable)

        def prefill(cache, ids):
            self.prefills += 1
            cache.tokens += ids

        def trim(cache, n):
            if self.trim_works:
                del cache.tokens[len(cache.tokens) - n:]

        self.pc = PrefixCache(make_cache=make_cache, prefill=prefill, offset=lambda c: len(c.tokens), trim=trim,
                              can_trim=lambda c: c.trimmable, enabled=enabled)

    def gen(self, ids, cache):
        context = (list(cache.tokens) if cache is not None else []) + list(ids)
        self.seen.append(context)
        if cache is not None:
            cache.tokens += list(ids) + ["<generated>"] * 3    # the cache grows while generating
        return "out:" + "|".join(context)

    def call(self, messages, scope=("model", "adapter"), mode="dictation_ft", raise_in_gen=False):
        parts = split_prompt(render, messages)
        prefix, full = parts if parts else (None, render(messages, True))
        gen = self.gen
        if raise_in_gen:
            def gen(ids, cache):
                self.gen(ids, cache)
                raise RuntimeError("boom")
        return self.pc.run(mode, scope, prefix, full, encode, gen)

    def plain(self, messages):
        return "out:" + "|".join(encode(render(messages, True)))


# ---- the split -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("examples", [(), (("a b", "A b."),), (("a b", "A b."), ("c d", "C d."))])
def test_split_falls_on_the_start_of_the_last_user_turn(examples):
    m = msgs(user="the real transcript", examples=examples)
    prefix, full = split_prompt(render, m)
    rest = full[len(prefix):]
    assert full.startswith(prefix) and prefix + rest == full
    assert prefix.endswith("<|eot|>") and "the real transcript" not in prefix
    assert rest == "<|user|>the real transcript<|eot|><|assistant|>"


def test_split_refuses_when_it_cannot_split_cleanly():
    stamps = iter(["day 1", "day 2"])         # a template that stamps the date, midnight between the renders
    moving = lambda m, g: f"[{next(stamps)}]" + render(m, g)
    assert split_prompt(moving, msgs()) is None
    assert split_prompt(render, [{"role": "user", "content": "x"}]) is None      # nothing before the last turn


def test_tokenized_prefix_plus_suffix_equals_tokenized_full_prompt():
    prefix, full = split_prompt(render, msgs(examples=(("a b", "A b."),)))
    assert encode(prefix) + encode(full[len(prefix):]) == encode(full)


def test_a_boundary_that_changes_the_tokens_runs_uncached():
    h = Harness()
    full = "<|bos|><|system|>foobar<|eot|>"
    out = h.pc.run("m", ("s",), "<|bos|><|system|>foo", full, encode, h.gen)     # "foo"+"bar" merge differently
    assert out == "out:" + "|".join(encode(full))
    assert h.pc.miss_reasons["token_mismatch"] == 1 and h.prefills == 0


# ---- hits, restore, no leak ------------------------------------------------------------------------------
def test_cached_output_equals_uncached_output_and_each_miss_logs_one_line(caplog):
    caplog.set_level(logging.DEBUG, LOG)
    h = Harness()
    for user in ["one", "two words here", "x", "one"]:
        m = msgs(user=user, examples=(("a b", "A b."),))
        assert h.call(m) == h.plain(m)
    assert (h.pc.hits, h.pc.misses) == (3, 1) and h.pc.miss_reasons["first_use"] == 1 and h.prefills == 1
    h.call(msgs(system="changed"))
    assert len([r for r in caplog.records if r.name == LOG]) == h.pc.misses == 2


def test_cache_is_restored_to_the_prefix_after_each_call_and_nothing_leaks():
    h = Harness()
    a = msgs(user="utterance AAA")
    b = msgs(user="utterance BBB")
    h.call(a)
    entry = h.pc._entries["dictation_ft"]
    assert len(entry.cache.tokens) == entry.n_prefix                   # restored
    h.call(b)
    assert "AAA" not in "".join(h.seen[1])                              # B never saw A
    assert h.seen[1][:entry.n_prefix] == encode(split_prompt(render, b)[0])


def test_an_exception_in_generation_still_restores_the_cache():
    h = Harness()
    with pytest.raises(RuntimeError):
        h.call(msgs(user="will fail"), raise_in_gen=True)
    entry = h.pc._entries["dictation_ft"]
    assert len(entry.cache.tokens) == entry.n_prefix
    assert h.call(msgs(user="after")) == h.plain(msgs(user="after")) and h.prefills == 1


# ---- invalidation ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("changed", [
    dict(system="a different system prompt"),
    dict(examples=(("new in", "New out."),)),
    dict(system="clean it up\n\nThese names are spelled exactly like this: Gauttam."),     # dictionary hint
])
def test_prefix_changes_rebuild_the_cache(changed):
    h = Harness()
    h.call(msgs())
    m = msgs(**changed)
    assert h.call(m) == h.plain(m)
    assert h.prefills == 2 and h.pc.miss_reasons["prefix_changed"] == 1
    h.call(m)
    assert h.prefills == 2                                              # warm again


@pytest.mark.parametrize("scope", [("other model", "adapter"), ("model", "personal adapter"), ("model", None)])
def test_model_or_adapter_changes_rebuild_the_cache(scope):
    h = Harness()
    h.call(msgs())
    assert h.call(msgs(), scope=scope) == h.plain(msgs())
    assert h.prefills == 2 and h.pc.miss_reasons["prefix_changed"] == 1


# ---- threads ---------------------------------------------------------------------------------------------
def test_a_cache_built_on_another_thread_is_discarded_and_rebuilt_here(caplog):
    h = Harness()
    caplog.set_level(logging.DEBUG, LOG)
    t = threading.Thread(target=lambda: h.call(msgs()))
    t.start(); t.join()
    assert h.prefills == 1
    assert h.call(msgs(user="now on this thread")) == h.plain(msgs(user="now on this thread"))
    assert h.prefills == 2 and h.pc.miss_reasons["thread_rebuild"] == 1
    assert "thread_rebuild" in caplog.text
    h.call(msgs())
    assert h.prefills == 2 and h.pc.hits == 1                           # cached again, on this thread


# ---- no silent fallbacks ---------------------------------------------------------------------------------
def test_disabled_and_unsplittable_prompts_run_uncached_and_say_why(caplog):
    caplog.set_level(logging.DEBUG, LOG)
    h = Harness(enabled=False)
    assert h.call(msgs()) == h.plain(msgs())
    assert h.pc.miss_reasons["disabled"] == 1 and h.prefills == 0 and h.pc.hits == 0
    h, full = Harness(), render(msgs(), True)
    assert h.pc.run("m", ("s",), None, full, encode, h.gen) == "out:" + "|".join(encode(full))
    assert h.pc.miss_reasons["no_prefix_match"] == 1
    assert "disabled" in caplog.text and "no_prefix_match" in caplog.text


def test_non_trimmable_cache_runs_uncached_builds_once_and_says_why(caplog):
    h = Harness(trimmable=False)
    caplog.set_level(logging.DEBUG, LOG)
    for _ in range(3):
        assert h.call(msgs()) == h.plain(msgs())
    assert h.pc.miss_reasons["non_trimmable"] == 3 and h.made == 1 and h.prefills == 0
    assert "non_trimmable" in caplog.text


def test_failed_trim_drops_the_cache_logs_and_the_next_call_is_correct(caplog):
    h = Harness(trim_works=False)
    caplog.set_level(logging.DEBUG, LOG)
    h.call(msgs())
    h.call(msgs(user="second"))                                         # first_use again: the entry was dropped
    h.trim_works = True
    assert h.call(msgs(user="third")) == h.plain(msgs(user="third"))
    assert h.pc.miss_reasons["trim_failed"] >= 1 and "trim_failed" in caplog.text
    assert h.seen[1] == encode(render(msgs(user="second"), True))       # nothing from call 1 leaked into call 2


# ---- wiring ----------------------------------------------------------------------------------------------
def test_prefix_cache_is_on_by_default_in_config():
    from dictum.config import Config
    assert Config.load().refine["mlx"]["prefix_cache"] is True


def test_real_model_cached_equals_uncached():
    """Needs MLX (skips itself off macOS) and the base model, which dictum cannot run without."""
    pytest.importorskip("mlx_lm")
    model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    from dictum.config import Config
    from dictum.refine.mlx_refiner import MLXRefiner
    cfg = Config.load()
    mode = cfg.mode("dictation_ft")
    kw = dict(model=model, adapter_path=None, adapter_fallback=None, use_personal=False, dictionary=cfg.dictionary)
    on, off = MLXRefiner(prefix_cache=True, **kw), MLXRefiner(prefix_cache=False, **kw)
    texts = ["um so I think we should uh ship it friday", "send the report to maria", "no wait monday", "ok thanks"]
    for t in texts:
        assert on._generate(mode, t, 60) == off._generate(mode, t, 60)
    assert on.cache_hits == len(texts) - 1 and on.cache_misses == 1
    assert off.cache_hits == 0 and off.cache_misses == len(texts)
