import numpy as np
import pytest

from dictum.config import Config
from dictum.pipeline import Pipeline
from dictum.types import AudioClip, Refined, Transcript

MULTI = "first paragraph here\n\nsecond paragraph here\n"


class FakeSTT:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def load(self): ...

    def transcribe(self, clip):
        return Transcript(text=self.text, engine="fake", latency_s=0.0, audio_s=clip.duration_s)


class FakeRefiner:
    name = "fake"

    def __init__(self, out=None, fail=False):
        self.out, self.fail = out, fail

    def warm_up(self, mode):
        return 0.0

    def refine(self, transcript, mode):
        if self.fail:
            raise RuntimeError("ollama down")
        return Refined(text=self.out, backend="fake", mode=mode.name, latency_s=0.0, source=transcript)


class FakeInjector:
    name = "fake"

    def __init__(self):
        self.pasted = []

    def inject(self, text):
        self.pasted.append(text)


CLIP = AudioClip(samples=np.full(16000, 0.1, dtype=np.float32), sample_rate=16000)


def make(stt_text, refiner):
    cfg = Config.load()
    cfg.raw["history"] = {"enabled": False}
    inj = FakeInjector()
    return Pipeline(None, FakeSTT(stt_text), refiner, inj, cfg), inj


# the three paths that end in inject: refined, one-word skip, refiner-failed fallback
PATHS = {
    "refined": lambda: make("send the draft", FakeRefiner(out=MULTI)),
    "one_word_skip": lambda: make("ls\n", FakeRefiner(out="unused")),
    "refiner_failed": lambda: make("echo hi\nrm -rf x\n", FakeRefiner(fail=True)),
}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("reason", ["terminal", "unknown"])
def test_terminal_target_pastes_one_line(path, reason):
    pipe, inj = PATHS[path]()
    refined, _ = pipe.process(CLIP, "dictation_ft", lambda: reason)
    (pasted,) = inj.pasted
    assert "\n" not in pasted and "\r" not in pasted and pasted == pasted.rstrip()
    assert refined.text == pasted
    assert len(refined.notes) == 1
    if reason == "unknown":
        assert "frontmost app unknown" in refined.notes[0]
    else:
        assert "terminal" in refined.notes[0]


@pytest.mark.parametrize("flatten", [None, lambda: None])
def test_other_apps_paste_unchanged(flatten):
    pipe, inj = PATHS["refined"]()
    refined, _ = pipe.process(CLIP, "dictation_ft", flatten)
    assert inj.pasted == [MULTI]
    assert refined.notes == []


def test_no_note_when_terminal_text_already_single_line():
    pipe, inj = make("send the draft", FakeRefiner(out="Send the draft."))
    refined, _ = pipe.process(CLIP, "dictation_ft", lambda: "terminal")
    assert inj.pasted == ["Send the draft."] and refined.notes == []


def test_history_records_the_pasted_text(monkeypatch):
    from dictum import history
    seen = []
    monkeypatch.setattr(history, "record", lambda raw, final, mode: seen.append(final) or 1)
    pipe, inj = PATHS["refined"]()
    pipe.cfg.raw["history"] = {"enabled": True}
    pipe.process(CLIP, "dictation_ft", lambda: "terminal")
    assert seen == inj.pasted and "\n" not in seen[0]
