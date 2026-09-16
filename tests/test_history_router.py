from pathlib import Path

from dictum import history
from dictum.config import Config, Mode
from dictum.refine import create_refiner


def test_history_roundtrip(tmp_path: Path):
    db = tmp_path / "h.db"
    i = history.record("um send it tuesday no thursday", "Send it Thursday.", "dictation", db=db)
    assert history.recent(1, db=db)[0].output == "Send it Thursday."
    assert history.training_pairs(db=db) == []
    history.correct(i, "Send it on Thursday.", db=db)
    assert history.training_pairs(db=db) == [{"id": i, "src": "um send it tuesday no thursday",
                                              "tgt": "Send it on Thursday."}]
    assert history.clear(db=db) == 1


def test_router_dispatches_by_mode():
    built = []

    class Fake:
        def __init__(self, name):
            self.name = name
            self.entries = None

        def refine(self, t, m):
            return self.name

        def warm_up(self, m):
            return 0.0

    def factory(cfg, dictionary, backend=None):
        built.append(backend)
        return Fake(backend)

    from dictum.refine.router import RouterRefiner
    r = RouterRefiner({"default_backend": "ollama"}, [], factory)
    assert r.refine(None, Mode(name="dictation_ft", backend="mlx")) == "mlx"
    assert r.refine(None, Mode(name="command")) == "ollama"
    assert r.refine(None, Mode(name="dictation_ft", backend="mlx")) == "mlx"
    assert built == ["mlx", "ollama"]


def test_adapter_resolve_missing_local_is_none():
    from dictum.adapters import resolve
    assert resolve("adapters/does-not-exist", use_personal=False) is None
    assert resolve(None, use_personal=False) is None
