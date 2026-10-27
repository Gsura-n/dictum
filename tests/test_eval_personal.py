"""Personal evaluation with synthetic WAVs and fake adapters; no hardware or models."""
import json
from itertools import count
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import yaml

from dictum.config import Config
from dictum.evals import command_score, normalize_numbers, wilson
from dictum.types import Refined, Transcript
from scripts import eval_personal as ep


def case(cid="P1", mode="dictation_ft", expect="250 samples"):
    return dict(id=cid, kind="read", mode=mode, said="two hundred and fifty samples", expect=expect)


class Refiner:
    name, adapter_path = "fake", "fake-adapter"
    output = "250 samples"

    def model_for(self, mode):
        return "fake-model"

    def warm_up(self, mode):
        if mode.name == "command":
            raise RuntimeError("backend unavailable")
        return 0

    def refine(self, source, mode):
        if mode.name == "command" and isinstance(self.output, Exception):
            raise self.output
        return Refined(self.output, self.name, mode.name, 0.5, source)


@pytest.fixture
def pipeline(monkeypatch):
    cfg = Config({"history": {"enabled": True}, "modes": {m: {} for m in ep.MODES}})
    stt = Mock(model_id="fake-stt", transcribe=lambda clip: Transcript("250 samples", "fake", 0.25, clip.duration_s))
    refiner = Refiner()
    monkeypatch.setattr("dictum.stt.create_engine", lambda *a, **kw: stt)
    monkeypatch.setattr("dictum.refine.create_refiner", lambda *a: refiner)
    monkeypatch.setattr("dictum.history.record", Mock(side_effect=AssertionError("history must be off")))
    monkeypatch.setattr("time.perf_counter", lambda: next(ticks))
    ticks = count()
    pipe = ep.build_pipeline(cfg)
    assert pipe.capture is None and pipe.injector.name == "null" and not cfg.raw["history"]["enabled"]
    return pipe


def audio(root, cid="P1", seconds=1, sr=16000, channels=1):
    path = root / "audio" / f"{cid}.wav"
    path.parent.mkdir(exist_ok=True)
    sf.write(path, np.full((int(seconds * sr), channels), 0.1), sr)


@pytest.mark.parametrize("mode,out,expect,passed", [
    ("dictation_ft", "250 samples", "250 samples", True),
    ("dictation_ft", "entirely different words", "250 samples", False),
    ("academic", "250 samples", "250\nsamples", False),
    ("academic", "250\nsamples", "250\nsamples", True),
    ("academic", "250\nsamples\n", "250\nsamples", False),
    ("command", " ls -la \n", ["ls -al", "ls -la"], True),
    ("command", "pwd", ["ls -al", "ls -la"], False),
])
def test_scores_real_pipeline_output(tmp_path, pipeline, mode, out, expect, passed):
    audio(tmp_path)
    pipeline.refiner.output = out
    (row,), models = ep.evaluate([case(mode=mode, expect=expect)], tmp_path, pipeline)
    assert ep.normalize_numbers is normalize_numbers and row["stt_wer"] == 0
    assert row["passed"] is passed and row["error"] is None and row["raw"] == "250 samples"
    assert (row["stt_s"], row["refine_s"], row["total_s"]) == (0.25, 0.5, 1)
    assert models[mode]["adapter"] == "fake-adapter" and models[mode]["model"] == "fake-model"
    if mode != "command":
        assert row["paste_similarity"] >= 0 and row["paste_wer"] == (1.5 if out.startswith("entirely") else 0)
    elif not passed:
        assert row["command_score"] == command_score(out, expect[0])


def test_backend_failure_is_failed_even_when_raw_matches_and_run_continues(tmp_path, pipeline):
    audio(tmp_path, "C1")
    audio(tmp_path)
    pipeline.refiner.output = RuntimeError("backend unavailable")
    original = pipeline.refiner.refine
    pipeline.refiner.refine = lambda src, mode: original(src, mode) if mode.name == "command" else Refined(
        src.text, "fake", mode.name, 0.5, src)
    rows, _ = ep.evaluate([case("C1", "command", ["250 samples"]), case()], tmp_path, pipeline)
    assert not rows[0]["passed"] and "backend unavailable" in rows[0]["error"] and rows[1]["passed"]
    assert rows[0]["stt_wer"] == 0 and rows[0]["refine_s"] > 0


def test_stt_and_audio_failures_do_not_stop_later_clips(tmp_path, pipeline):
    audio(tmp_path, "P2")
    audio(tmp_path, "P3")
    pipeline.stt.transcribe = Mock(side_effect=[RuntimeError("STT failed"), Transcript("250 samples", "fake", .25, 1)])
    rows, _ = ep.evaluate([case("P1"), case("P2"), case("P3")], tmp_path, pipeline)
    assert [r["passed"] for r in rows] == [False, False, True]
    assert rows[0]["error"] and rows[1]["error"] and rows[1]["stt_wer"] is None


def manifest(root, clips=None, refs=None):
    (root / "clips.yaml").write_text(yaml.safe_dump({"clips": clips if clips is not None else [case()]}))
    (root / "references.yaml").write_text(yaml.safe_dump(refs if refs is not None else {"P1": case()}))


@pytest.mark.parametrize("clips,refs", [([], {}), ([case(), case()], {}), ([None], {}), ([case()], []),
    ([case("../P1")], {}), ([case(mode="other")], {}), ([dict(case(), kind="other")], {}),
    ([case()], {}), ([case()], {"P1": {"said": 3, "expect": "x"}}),
    ([case()], {"P1": {"said": "x", "expect": ["x"]}}),
    ([case(mode="command")], {"P1": {"said": "x", "expect": []}}),
])
def test_invalid_inputs_rejected(tmp_path, clips, refs):
    manifest(tmp_path, clips, refs)
    with pytest.raises(ValueError):
        ep.load_suite(tmp_path)


def test_group_and_overall_math():
    rows = [dict(id=cid, passed=ok, error=error, stt_wer=w, audio_s=duration,
                 stt_s=t, refine_s=2*t, total_s=3*t)
            for cid, ok, error, w, duration, t in [("P1", True, None, .2, 15, 1),
                ("P2", False, None, .4, 16, 3), ("C1", False, "failed", .6, 10, 100)]]
    summary = ep.summarize(rows)
    s = summary["overall"]
    assert s["n"] == 3 and s["mean_stt_wer"] == pytest.approx(.4) and s["paste_pass_rate"] == 1/3
    assert s["paste_ci95"] == pytest.approx(wilson(1, 3))
    assert s["latency"]["stt_s"] == {"p50": 2, "p95": pytest.approx(2.9)}
    assert s["latency_le15s"]["total_s"] == {"p50": 3, "p95": 3}
    assert summary["P"]["paste_pass_rate"] == .5 and summary["C"]["failed"] == 1
    assert summary["Q"]["n"] == 0 and summary["Q"]["mean_stt_wer"] is None
    for n in range(len(rows) + 1):
        s = ep.summarize(rows[:n])
        assert s["overall"]["n"] == sum(s[g]["n"] for g in ep.GROUPS) == n
        assert s["overall"]["paste_pass_rate"] == (sum(r["passed"] for r in rows[:n])/n if n else None)


def test_main_filters_saves_and_prints_models_and_latency(tmp_path, pipeline, monkeypatch, capsys):
    clips = [case(), case("A1", "academic")]
    manifest(tmp_path, clips, {c["id"]: c for c in clips})
    audio(tmp_path)
    monkeypatch.setattr(ep.Config, "load", lambda: pipeline.cfg)
    monkeypatch.setattr(ep, "build_pipeline", lambda cfg: pipeline)
    monkeypatch.setattr("sys.argv", ["eval_personal", "--dir", str(tmp_path), "--only", "P1,A1", "--mode", "dictation_ft"])
    ep.main()
    (out,) = (tmp_path / "results").glob("*.json")
    data = json.loads(out.read_text())
    assert [r["id"] for r in data["rows"]] == ["P1"] and data["summary"]["overall"]["paste_pass_rate"] == 1
    shown = capsys.readouterr().out
    assert all(s in shown for s in ("fake-model", "fake-adapter", "STT WER", "overall", "<=15s", "p50", "p95"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio", "clips.yaml", "references.yaml", "results"]


def test_refuses_repo_and_results_symlink_into_repo(tmp_path, monkeypatch):
    for root in (ep.ROOT, tmp_path):
        if root == tmp_path:
            (root / "results").symlink_to(ep.ROOT, target_is_directory=True)
        monkeypatch.setattr("sys.argv", ["eval_personal", "--dir", str(root)])
        with pytest.raises((ValueError, SystemExit), match="repo"):
            ep.main()


@pytest.mark.parametrize("phase", ["startup", "metadata"])
def test_setup_failures_are_recorded(tmp_path, pipeline, phase):
    audio(tmp_path)
    target, attr = (pipeline.stt, "load") if phase == "startup" else (pipeline.refiner, "model_for")
    setattr(target, attr, Mock(side_effect=RuntimeError("unavailable")))
    (row,), models = ep.evaluate([case()], tmp_path, pipeline)
    assert ("unavailable" in row["error"]) if phase == "startup" else ("unavailable" in models["dictation_ft"]["error"])


@pytest.mark.parametrize("sr,channels", [(8000, 1), (16000, 2)])
def test_wrong_audio_format_is_a_failed_row(tmp_path, pipeline, sr, channels):
    audio(tmp_path, sr=sr, channels=channels)
    (row,), _ = ep.evaluate([case()], tmp_path, pipeline)
    assert not row["passed"] and "16 kHz mono" in row["error"]


@pytest.mark.parametrize("only,mode", [("missing", None), (None, "command")])
def test_empty_or_unknown_selection_is_rejected(tmp_path, only, mode):
    manifest(tmp_path)
    with pytest.raises(ValueError):
        ep.load_suite(tmp_path, only, mode)
