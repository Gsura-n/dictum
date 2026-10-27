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


def test_history_training_pairs_edge_cases(tmp_path: Path):
    db = tmp_path / "h2.db"

    # 1. Ignore uncorrected records
    i1 = history.record("uncorrected transcript", "uncorrected output", "dictation", db=db)

    # 2. Modes filtering
    i2 = history.record("wrong mode transcript", "wrong mode output", "command", db=db)
    history.correct(i2, "wrong mode corrected", db=db)

    # 3. Valid records for ordering and mode matching
    i3 = history.record("t2", "o2", "dictation", db=db)
    history.correct(i3, "c2", db=db)

    i4 = history.record("t1", "o1", "dictation_ft", db=db)
    history.correct(i4, "c1", db=db)

    pairs = history.training_pairs(db=db)

    # Should only contain i3 and i4, ordered by id
    assert len(pairs) == 2
    assert pairs[0]["id"] == i3
    assert pairs[0]["src"] == "t2"
    assert pairs[0]["tgt"] == "c2"
    assert pairs[1]["id"] == i4

    # Specific mode filtering
    assert len(history.training_pairs(modes=("dictation",), db=db)) == 1
    assert history.training_pairs(modes=("dictation",), db=db)[0]["id"] == i3



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
