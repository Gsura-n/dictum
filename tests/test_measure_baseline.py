import builtins
import importlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts import measure_baseline as baseline


def cached(hits=40, misses=1, **reasons):
    return SimpleNamespace(prefix_cache=SimpleNamespace(
        hits=hits, misses=misses, miss_reasons=Counter(reasons)))


@pytest.mark.parametrize("refiner", [object(), SimpleNamespace(_backends={})])
def test_prefix_cache_stats_without_cache(refiner):
    assert baseline.prefix_cache_stats(refiner) is None


@pytest.mark.parametrize("hits,misses", [(0, 0), (0, 1), (40, 1), (3, 7)])
def test_prefix_cache_stats_direct(hits, misses):
    refiner = cached(hits, misses, first_use=misses)
    assert baseline.prefix_cache_stats(refiner) == {
        "hits": hits, "misses": misses, "miss_reasons": {"first_use": misses}}


def test_prefix_cache_stats_router():
    router = SimpleNamespace(_backends={"mlx": cached(first_use=1), "plain": object()})
    assert baseline.prefix_cache_stats(router) == {
        "hits": 40, "misses": 1, "miss_reasons": {"first_use": 1}}


def test_prefix_cache_stats_sums_reachable_refiners_once():
    child = cached(2, 2, first_use=1, thread_rebuild=1)
    router = cached(3, 1, first_use=1)
    nested = SimpleNamespace(_backends={"mlx": child, "parent": router})
    router._backends = {"nested": nested, "shared": child}
    assert baseline.prefix_cache_stats(router) == {
        "hits": 5, "misses": 3,
        "miss_reasons": {"first_use": 2, "thread_rebuild": 1}}
    assert child.prefix_cache.miss_reasons == {"first_use": 1, "thread_rebuild": 1}


def test_import_does_not_load_audio_dependencies(monkeypatch):
    original = builtins.__import__

    def without_audio(name, *args, **kwargs):
        assert name not in {"soundfile", "soxr"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_audio)
    assert callable(importlib.reload(baseline).prefix_cache_stats)


@pytest.mark.parametrize("with_cache", [False, True])
def test_main_reports_prefix_cache_after_processing(monkeypatch, tmp_path, capsys, with_cache):
    refiner = cached(0, 0) if with_cache else object()
    router = SimpleNamespace(_backends={})
    cfg = SimpleNamespace(raw={}, refine={}, dictionary=[],
                          stt_engine_config=lambda _: ("fake", {}))
    monkeypatch.setattr(baseline.Config, "load", lambda: cfg)
    monkeypatch.setattr("dictum.refine.create_refiner", lambda *args: router)
    monkeypatch.setattr("dictum.stt.create_engine", lambda *args, **kwargs: object())

    class FakePipeline:
        def __init__(self, **kwargs):
            self.refiner = kwargs["refiner"]

        def warm_up(self, **kwargs):
            return {}

        def process(self, clip, mode):
            self.refiner._backends["fake"] = refiner
            if with_cache:
                refiner.prefix_cache.hits = 40
                refiner.prefix_cache.misses = 1
                refiner.prefix_cache.miss_reasons["first_use"] = 1
            return (SimpleNamespace(backend="fake", notes=[], text="Example."),
                    SimpleNamespace(audio_s=10, stt_s=0.1, refine_s=0.2, total_s=0.3))

    monkeypatch.setattr(baseline, "Pipeline", FakePipeline)
    monkeypatch.setattr(baseline, "utterances", lambda *args: [(object(), "Example.")])
    monkeypatch.setattr(baseline, "memory", lambda: {
        "rss_mb": 10, "peak_rss_mb": 12, "footprint": "fake footprint"})
    monkeypatch.setattr(baseline, "machine", lambda: {
        "chip": "fake", "memory_gb": 16, "commit": "fake", "dirty": False})
    out = tmp_path / "baseline.json"
    monkeypatch.setattr("sys.argv", ["measure_baseline.py", "--out", str(out)])
    baseline.main()

    stats = json.loads(out.read_text())["summary"]["prefix_cache"]
    output = capsys.readouterr().out
    if with_cache:
        assert stats == {"hits": 40, "misses": 1, "miss_reasons": {"first_use": 1}}
        line = "prefix cache: 40 hits, 1 miss (first_use: 1)"
        assert output.count(line) == 1
        assert output.index("  footprint") < output.index(line) < output.index("saved ")
    else:
        assert stats is None
        assert "prefix cache:" not in output
