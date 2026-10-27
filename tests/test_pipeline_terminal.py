from unittest.mock import Mock

import numpy as np
import pytest

from dictum import dictionary as dict_mod
from dictum.config import Config
from dictum.pipeline import Pipeline
from dictum.types import AudioClip, Refined, Transcript

MULTI = "first paragraph here\n\nsecond paragraph here\n"


class FakeSTT:
    name = "fake"

    def __init__(self, text, latency_s=0.0):
        self.text = text
        self.latency_s = latency_s

    def load(self): ...

    def transcribe(self, clip):
        return Transcript(text=self.text, engine="fake", latency_s=self.latency_s, audio_s=clip.duration_s)


class FakeRefiner:
    name = "fake"

    def __init__(self, out=None, fail=False, latency_s=0.0):
        self.out, self.fail = out, fail
        self.latency_s = latency_s
        self.calls = []

    def warm_up(self, mode):
        return 0.0

    def refine(self, transcript, mode):
        self.calls.append((transcript, mode))
        if isinstance(self.fail, Exception):
            raise self.fail
        if self.fail:
            raise RuntimeError("ollama down")
        text = self.out[mode.name] if isinstance(self.out, dict) else self.out
        return Refined(text=text, backend="fake", mode=mode.name,
                       latency_s=self.latency_s, source=transcript)


class FakeInjector:
    name = "fake"

    def __init__(self):
        self.pasted = []

    def inject(self, text):
        self.pasted.append(text)


CLIP = AudioClip(samples=np.full(16000, 0.1, dtype=np.float32), sample_rate=16000)


def make(stt_text, refiner, cfg=None):
    cfg = Config.load() if cfg is None else cfg
    cfg.raw["history"] = {"enabled": False}
    inj = FakeInjector()
    return Pipeline(None, FakeSTT(stt_text), refiner, inj, cfg), inj


@pytest.fixture(autouse=True)
def history_record(monkeypatch):
    record = Mock(return_value=1)
    monkeypatch.setattr("dictum.history.record", record)
    return record


@pytest.fixture
def contract_config():
    return Config(raw={
        "default_mode": "dictation_ft",
        "modes": {
            "dictation_ft": {},
            "command": {"max_input_words": 4, "fallback_mode": "dictation_ft"},
        },
    })


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


def test_terminal_target_strips_unsafe_text_to_empty():
    pipe, inj = make("some words\n", FakeRefiner(out="\n\n\x1b"))
    pipe.cfg.raw["history"] = {"enabled": True}
    refined, _ = pipe.process(CLIP, "dictation_ft", lambda: "terminal")

    assert inj.pasted == ["some words"]
    assert refined.text == "some words"
    assert any("terminal" in note for note in refined.notes)
    assert any("output empty after flattening" in note for note in refined.notes)
    # history records the pasted text
    assert any("history" in note for note in refined.notes)


def test_terminal_target_strips_spaces_to_empty():
    pipe, inj = make("some words\n", FakeRefiner(out="   \n  \n "))
    pipe.cfg.raw["history"] = {"enabled": True}
    refined, _ = pipe.process(CLIP, "dictation_ft", lambda: "terminal")

    assert inj.pasted == ["some words"]
    assert refined.text == "some words"
    assert any("terminal" in note for note in refined.notes)
    assert any("output empty after flattening" in note for note in refined.notes)
    # history records the pasted text
    assert any("history" in note for note in refined.notes)


def test_history_records_the_pasted_text(monkeypatch):
    from dictum import history
    seen = []
    monkeypatch.setattr(history, "record", lambda raw, final, mode: seen.append(final) or 1)
    pipe, inj = PATHS["refined"]()
    pipe.cfg.raw["history"] = {"enabled": True}
    pipe.process(CLIP, "dictation_ft", lambda: "terminal")
    assert seen == inj.pasted and "\n" not in seen[0]


@pytest.mark.parametrize("raw", ["", " \t\n", "hello", "hello\n"])
def test_process_skips_empty_and_one_word_transcripts(raw, contract_config):
    refiner = FakeRefiner(out="must not be used")
    pipe, inj = make(raw, refiner, contract_config)

    refined, _ = pipe.process(CLIP, "dictation_ft")

    assert refiner.calls == []
    assert inj.pasted == [raw]
    assert refined.text == raw
    assert refined.source.text == raw
    assert refined.backend == "skipped"


@pytest.mark.parametrize("entry_source", ["config", "refiner", "uninitialized", "empty"])
@pytest.mark.parametrize("raw", ["zorin", "hello", "", " \t\n"])
def test_process_skip_applies_dictionary_without_refining(raw, entry_source, contract_config):
    dictionary = [{"word": "Zoryn", "sounds_like": ["zorin"]}]
    contract_config.raw["dictionary"] = dictionary
    refiner = FakeRefiner(out="must not be used")
    if entry_source == "refiner":
        refiner.entries = dict_mod.parse(dictionary)
        # Eval overrides must take precedence over the configured dictionary.
        contract_config.raw["dictionary"] = [{"word": "Other", "sounds_like": ["zorin"]}]
    elif entry_source == "uninitialized":
        refiner.entries = None
    elif entry_source == "empty":
        refiner.entries = []
    pipe, inj = make(raw, refiner, contract_config)

    refined, _ = pipe.process(CLIP, "dictation_ft")

    changed = raw == "zorin" and entry_source != "empty"
    expected = "Zoryn" if changed else raw
    assert refiner.calls == []
    assert inj.pasted == [expected]
    assert refined.text == expected
    assert refined.source.text == raw
    assert refined.backend == "skipped"
    assert refined.notes == (["dictionary applied"] if changed else [])


@pytest.mark.parametrize("error", [RuntimeError("model unavailable"), TimeoutError("timed out")])
def test_process_refiner_error_injects_raw_transcript(error, contract_config):
    raw = "um send the draft\nplease"
    refiner = FakeRefiner(fail=error)
    pipe, inj = make(raw, refiner, contract_config)

    refined, _ = pipe.process(CLIP, "dictation_ft")

    assert len(refiner.calls) == 1
    assert inj.pasted == [raw]
    assert refined.text == raw
    assert refined.source.text == raw
    assert refined.backend.startswith("failed")
    assert type(error).__name__ in refined.backend


@pytest.mark.parametrize("enabled, fails", [(False, False), (True, False), (True, True)])
def test_process_history_contract(enabled, fails, contract_config, history_record):
    raw, final = "um send the draft", "Send the draft."
    pipe, inj = make(raw, FakeRefiner(out=final), contract_config)
    pipe.cfg.raw["history"]["enabled"] = enabled
    if fails:
        history_record.side_effect = OSError("disk unavailable")

    refined, _ = pipe.process(CLIP, "dictation_ft")

    assert inj.pasted == [final]
    assert refined.text == final
    if enabled:
        history_record.assert_called_once_with(raw, final, "dictation_ft")
    else:
        history_record.assert_not_called()
    if fails:
        assert any("history not saved" in note for note in refined.notes)
    else:
        assert not any("history not saved" in note for note in refined.notes)


@pytest.mark.parametrize("entrypoint", ["process", "refine_text"])
@pytest.mark.parametrize("raw, expected_mode", [
    ("list these files", "command"),
    ("please list these files", "command"),
    ("please list these files now", "dictation_ft"),
])
def test_max_input_words_fallback(raw, expected_mode, entrypoint, contract_config):
    outputs = {"command": "echo files", "dictation_ft": "Please list these files now."}
    refiner = FakeRefiner(out=outputs)
    pipe, inj = make(raw, refiner, contract_config)
    transcript = Transcript(text=raw, engine="fake", latency_s=0.0, audio_s=CLIP.duration_s)

    if entrypoint == "process":
        refined, _ = pipe.process(CLIP, "command")
        assert inj.pasted == [outputs[expected_mode]]
    else:
        refined = pipe.refine_text(transcript, "command")
        assert inj.pasted == []

    assert refiner.calls == [(transcript, contract_config.mode(expected_mode))]
    assert refined.text == outputs[expected_mode]
    assert refined.mode == expected_mode
    assert refined.source == transcript
    if expected_mode == "dictation_ft":
        assert any("too long" in note and "dictation_ft" in note for note in refined.notes)
    else:
        assert refined.notes == []


@pytest.mark.parametrize("entrypoint", ["process", "refine_text"])
def test_command_prose_falls_back_to_dictation(entrypoint, contract_config):
    raw = "list these files"
    # A multiline response fails looks_like_command regardless of executables on PATH.
    prose = "Here are the files.\nThey are listed below."
    outputs = {"command": prose, "dictation_ft": "List these files."}
    refiner = FakeRefiner(out=outputs)
    pipe, inj = make(raw, refiner, contract_config)
    transcript = Transcript(text=raw, engine="fake", latency_s=0.0, audio_s=CLIP.duration_s)

    if entrypoint == "process":
        refined, _ = pipe.process(CLIP, "command")
        assert inj.pasted == [outputs["dictation_ft"]]
    else:
        refined = pipe.refine_text(transcript, "command")
        assert inj.pasted == []

    assert refiner.calls == [
        (transcript, contract_config.mode("command")),
        (transcript, contract_config.mode("dictation_ft")),
    ]
    assert refined.text == outputs["dictation_ft"]
    assert refined.mode == "dictation_ft"
    assert refined.source == transcript
    assert any("not a runnable command" in note and "dictation_ft" in note for note in refined.notes)


@pytest.mark.parametrize("raw, fail, refine_s", [
    ("send the draft", False, 0.25),
    ("hello", False, 0.0),
    ("", False, 0.0),
    ("send the draft", True, 0.0),
])
def test_process_fills_timing_fields(raw, fail, refine_s, contract_config, monkeypatch):
    ticks = iter([10.0, 11.0])
    monkeypatch.setattr("dictum.pipeline.time.perf_counter", lambda: next(ticks))
    pipe, _ = make(raw, FakeRefiner(out="Send the draft.", fail=fail, latency_s=0.25),
                   contract_config)
    pipe.stt = FakeSTT(raw, latency_s=0.125)

    _, timing = pipe.process(CLIP, "dictation_ft")

    assert timing.audio_s == CLIP.duration_s
    assert timing.stt_s == 0.125
    assert timing.refine_s == refine_s
    assert timing.total_s == 1.0
    assert timing.total_s >= timing.refine_s
