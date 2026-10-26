"""Prompt lookup decoding, tested without MLX. The fake model's next token is a function of the whole cache, so
a wrongly trimmed or leaked cache changes the output. Greedy decoding with one token per forward is the reference."""
import random

import pytest
import yaml

from dictum.config import DEFAULT_PATH
from dictum.refine.lookup_decode import LookupStats, draft, lookup_generate
from dictum.refine.prefix_cache import PrefixCache


class FakeLM:
    """cache is the list of tokens fed so far; forward feeds tokens and returns the last n_out predictions."""

    def __init__(self, rule, fail_on_call=None):
        self.rule, self.cache, self.calls, self.fail_on_call = rule, [], 0, fail_on_call

    def forward(self, tokens, n_out):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("boom")
        preds = []
        for t in tokens:
            self.cache.append(t)
            preds.append(self.rule(self.cache))
        return preds[-n_out:]

    def trim(self, n):
        del self.cache[len(self.cache) - n:]


def greedy(rule, ids, max_tokens, eos):
    """One token per forward, stopping like mlx_lm.stream_generate: on EOS (not emitted) or at max_tokens."""
    lm, out = FakeLM(rule), []
    tok = lm.forward(ids, 1)[0]
    while True:
        out.append(tok)
        if tok in eos or len(out) >= max_tokens:
            break
        tok = lm.forward([tok], 1)[0]
    return out[:-1] if out[-1] in eos else out


def lookup(rule, ids, max_tokens, eos, ngram=2, k=4, stats=None):
    lm = FakeLM(rule)
    out = lookup_generate(lm.forward, lm.trim, ids, max_tokens=max_tokens, eos=eos, ngram=ngram,
                          draft_tokens=k, stats=stats)
    return out, lm


def mixed(h):                  # depends on three tokens, so a two-token lookup is right only some of the time
    a, b, c = ([0, 0] + h)[-1], ([0, 0] + h)[-2], ([0, 0] + h)[-3]
    return 1 + (a * 5 + b * 3 + c) % 9


# ---- drafting ---------------------------------------------------------------------------------------------
def test_draft_takes_the_latest_earlier_match():
    ctx = [1, 2, 3, 9, 1, 2, 4, 7, 1, 2]
    assert draft(ctx, 2, 2) == [4, 7]          # not [3, 9], the older occurrence of 1 2


def test_draft_is_cut_by_k_and_by_the_end_of_the_context():
    assert draft([1, 2, 5, 6, 7, 8, 1, 2], 2, 3) == [5, 6, 7]
    assert draft([1, 2, 5, 1, 2], 2, 4) == [5, 1, 2]


@pytest.mark.parametrize("ctx", [[1, 2, 3, 4], [7], [], [1, 2]])
def test_draft_is_empty_without_an_earlier_match(ctx):
    assert draft(ctx, 2, 4) == []


def test_draft_matches_overlapping_earlier_occurrences_only():
    assert draft([3, 4, 3, 4], 2, 4) == [3, 4]  # an earlier 3 4 may overlap the end; the end itself is not a match
    assert draft([1, 1, 1], 2, 4) == [1]


# ---- the loop ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_identical_to_plain_greedy(seed):
    rnd = random.Random(seed)
    ids = [rnd.randint(1, 9) for _ in range(rnd.randint(3, 30))]
    max_tokens, eos = rnd.randint(1, 40), {9} if seed % 2 else set()
    ngram, k = rnd.choice([1, 2, 3]), rnd.choice([1, 2, 4, 6])
    got, _ = lookup(mixed, ids, max_tokens, eos, ngram, k)
    assert got == greedy(mixed, ids, max_tokens, eos)


def test_drafts_are_sometimes_accepted_and_sometimes_rejected():
    stats = LookupStats()
    for seed in range(40):
        rnd = random.Random(seed)
        lookup(mixed, [rnd.randint(1, 9) for _ in range(20)], 30, set(), stats=stats)
    assert 0 < stats.accepted < stats.drafted and stats.steps > 0


def test_cache_holds_exactly_the_committed_tokens_after_rejections():
    ids = [3, 1, 4, 1, 5, 8, 2, 6, 5, 3, 5]
    out, lm = lookup(mixed, ids, 25, set())
    assert len(out) == 25
    fed = (ids + out)[:len(lm.cache)]          # what is left in the cache is what was emitted, nothing rejected
    assert fed == lm.cache and len(lm.cache) >= len(ids) + len(out) - 1


def test_draft_is_cut_at_max_tokens():
    period8 = lambda h: h[len(h) - 8] if len(h) >= 8 else 0  # noqa: E731
    ids = list(range(1, 9)) * 3
    out, lm = lookup(period8, ids, 3, set(), k=4)
    assert out == [1, 2, 3]
    assert len(lm.cache) == len(ids) + 3             # a draft longer than the tokens left would have been fed too


def test_eos_inside_a_draft_stops_there():
    copy = lambda h: h[len(h) - 4] if len(h) >= 4 else 0             # repeats the prompt 4 5 6 9, and 9 is EOS  # noqa: E731
    stats = LookupStats()
    out, _ = lookup(copy, [4, 5, 6, 9], 20, {9}, stats=stats)
    assert out == [4, 5, 6] == greedy(copy, [4, 5, 6, 9], 20, {9})
    assert stats.accepted >= 2                 # 6 and the EOS came from the draft


def test_eos_as_the_first_token_gives_nothing():
    out, _ = lookup(lambda h: 9, [1, 2, 3], 10, {9})
    assert out == []


# ---- with the prefix cache --------------------------------------------------------------------------------
def build(rule, fail_on_call=None):
    made = []

    def make():
        made.append(FakeLM(rule, fail_on_call))
        return made[-1]

    pc = PrefixCache(make_cache=make, prefill=lambda c, ids: c.forward(ids, 1), offset=lambda c: len(c.cache),
                     trim=lambda c, n: c.trim(n), can_trim=lambda c: True)

    def run(prefix, tail):
        def gen(ids, cache):
            return lookup_generate(cache.forward, cache.trim, ids, max_tokens=12, eos=set(), ngram=2, draft_tokens=4)
        enc = lambda s: [ord(ch) % 9 + 1 for ch in s]  # noqa: E731
        return pc.run("m", ("model", None), prefix, prefix + tail, enc, gen)

    return pc, made, run


def test_offset_is_back_at_the_prefix_after_every_call():
    pc, made, run = build(mixed)
    first = run("system", "abcabcab")
    assert len(made[0].cache) == len("system")
    run("system", "xyzxyzx")
    assert len(made[0].cache) == len("system")
    assert run("system", "abcabcab") == first  # same input, same output: nothing leaked from the call between
    assert pc.hits == 2 and len(made) == 1


def test_offset_is_back_at_the_prefix_after_an_exception():
    pc, made, run = build(mixed, fail_on_call=3)   # call 1 prefills, 2 is the first forward, 3 raises
    with pytest.raises(RuntimeError, match="boom"):
        run("system", "abcabcab")
    assert len(made[0].cache) == len("system")


# ---- config and the real model ----------------------------------------------------------------------------
def test_config_defaults():
    mlx = yaml.safe_load(DEFAULT_PATH.read_text())["refine"]["mlx"]
    assert (mlx["lookup_decoding"], mlx["lookup_ngram"], mlx["lookup_draft_tokens"]) == (True, 2, 4)


def test_real_model_lookup_equals_plain():
    """Needs MLX (skips itself off macOS) and the base model, which dictum cannot run without."""
    pytest.importorskip("mlx_lm")
    from dictum.config import Config
    from dictum.refine.mlx_refiner import MLXRefiner
    cfg = Config.load()
    mode = cfg.mode("dictation_ft")
    kw = dict(model="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path=None, adapter_fallback=None,
              use_personal=False, dictionary=cfg.dictionary)
    on, off = MLXRefiner(lookup_decoding=True, **kw), MLXRefiner(lookup_decoding=False, **kw)
    for t in ["um so I think we should uh ship it friday", "send the report to maria", "no wait monday", "ok thanks"]:
        assert on._generate(mode, t, 60) == off._generate(mode, t, 60)
    assert on.lookup_stats.steps > 0 and off.lookup_stats.steps == 0
