"""scripts/record_personal.py with a fake mic, a fake STT engine and scripted keystrokes: no hardware, no model.
Every take is a different constant, so the saved wav shows which take was kept."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from dictum.types import AudioClip, Transcript
from scripts import record_personal as rp

REPO = Path(rp.__file__).resolve().parents[1]
ENTER = ""


class FakeCapture:
    def __init__(self, rate=16000):
        self.rate, self.takes = rate, 0

    def start(self):
        self.takes += 1

    def stop(self):
        return AudioClip(samples=np.full(self.rate, self.takes / 10, dtype=np.float32), sample_rate=self.rate)


class FakeSTT:
    def transcribe(self, clip):
        return Transcript(text=f"heard take {round(float(clip.samples[0]) * 10)}", engine="fake", latency_s=0.0,
                          audio_s=len(clip.samples) / clip.sample_rate)


def keys(*answers):
    it = iter(answers)

    def ask(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    return ask


def clip(id, kind="read", mode="dictation_ft", **kw):
    base = {"id": id, "kind": kind, "mode": mode}
    return {**base, **({"say": f"say {id}", "expect": f"expect {id}"} if kind == "read" else {"cue": f"cue {id}"}), **kw}


CLIPS = [clip("A1"), clip("A2", "cue"), clip("A3")]
TAKE = [ENTER, ENTER]            # start, stop


def run(root, *answers, clips=CLIPS, capture=None, **kw):
    shown = []
    status = rp.record_session(clips, root, capture or FakeCapture(), FakeSTT(), keys(*answers), shown.append, **kw)
    return status, "\n".join(shown)


def lines(root):
    p = root / "transcripts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ---- the clip file -----------------------------------------------------------------------------------------
def write_clips(tmp_path, items):
    p = tmp_path / "clips.yaml"
    p.write_text(yaml.safe_dump({"clips": items}))
    return p


def test_load_clips_reads_the_list(tmp_path):
    assert rp.load_clips(write_clips(tmp_path, CLIPS)) == CLIPS


@pytest.mark.parametrize("bad, msg", [
    ([{"id": "X", "kind": "sing", "mode": "m", "cue": "c"}], "kind"),
    ([{"id": "X", "kind": "read", "mode": "m", "say": "s"}], "expect"),
    ([{"id": "X", "kind": "cue", "mode": "m"}], "cue"),
    ([{"id": "../X", "kind": "cue", "mode": "m", "cue": "c"}], "id"),
    ([clip("X"), clip("X")], "duplicate"),
])
def test_load_clips_rejects_a_bad_entry(tmp_path, bad, msg):
    with pytest.raises(ValueError, match=msg):
        rp.load_clips(write_clips(tmp_path, bad))


# ---- the session -------------------------------------------------------------------------------------------
def test_shows_the_read_text_or_the_cue_but_never_the_expected_text(tmp_path):
    _, out = run(tmp_path, *TAKE, ENTER, *TAKE, ENTER, "q")
    assert "A1" in out and "READ" in out and "say A1" in out
    assert "SPEAK FREELY" in out and "cue A2" in out
    assert "expect" not in out


def test_resume_starts_at_the_first_clip_without_audio(tmp_path):
    (tmp_path / "audio").mkdir()
    for done in ("A1", "A2"):
        sf.write(tmp_path / "audio" / f"{done}.wav", np.zeros(1600, np.float32), 16000)
    assert rp.first_missing(CLIPS, tmp_path) == 2
    status, out = run(tmp_path, *TAKE, ENTER)
    assert status == "done" and "A3" in out and "say A1" not in out
    assert [x["id"] for x in lines(tmp_path)] == ["A3"]


def test_a_quit_session_resumes_where_it_stopped(tmp_path):
    assert run(tmp_path, *TAKE, ENTER, "q")[0] == "quit"          # keeps A1, quits at A2
    assert run(tmp_path, *TAKE, ENTER, *TAKE, ENTER)[0] == "done"
    assert [x["id"] for x in lines(tmp_path)] == ["A1", "A2", "A3"]
    assert rp.first_missing(CLIPS, tmp_path) is None


def test_redo_overwrites_the_first_take(tmp_path):
    run(tmp_path, *TAKE, "r", *TAKE, ENTER, "q", clips=CLIPS[:1])
    samples, _ = sf.read(tmp_path / "audio" / "A1.wav")
    assert samples[0] == pytest.approx(0.2, abs=1e-3)              # take 2
    (row,) = lines(tmp_path)
    assert row["transcript"] == "heard take 2"


def test_skip_writes_nothing(tmp_path):
    run(tmp_path, *TAKE, "s", "q", clips=CLIPS[:2])
    assert not (tmp_path / "audio" / "A1.wav").exists()
    assert lines(tmp_path) == []


def test_transcripts_has_one_line_per_kept_clip(tmp_path):
    run(tmp_path, *TAKE, ENTER, *TAKE, "s", *TAKE, ENTER)          # A1 kept, A2 skipped, A3 kept
    rows = lines(tmp_path)
    assert [r["id"] for r in rows] == ["A1", "A3"]
    r = rows[0]
    assert (r["kind"], r["mode"], r["transcript"], r["audio_s"]) == ("read", "dictation_ft", "heard take 1", 1.0)
    assert r["timestamp"][:2] == "20" and set(r) == {"id", "kind", "mode", "transcript", "audio_s", "timestamp"}


def test_wav_is_16khz_mono_even_when_the_mic_is_not(tmp_path):
    run(tmp_path, *TAKE, ENTER, clips=CLIPS[:1], capture=FakeCapture(rate=48000))
    info = sf.info(tmp_path / "audio" / "A1.wav")
    assert (info.samplerate, info.channels) == (16000, 1)


def test_from_a_clip_redoes_it_and_replaces_its_line(tmp_path):
    run(tmp_path, *TAKE, ENTER, *TAKE, ENTER, *TAKE, ENTER)
    run(tmp_path, *TAKE, ENTER, "q", start_id="A2")                 # take 4 replaces A2
    rows = lines(tmp_path)
    assert [r["id"] for r in rows] == ["A1", "A3", "A2"]            # one line per clip, the new one last
    assert rows[-1]["transcript"] == "heard take 1"                 # the second session has its own FakeCapture


def test_ctrl_d_quits_without_keeping_the_take(tmp_path):
    assert run(tmp_path, *TAKE, clips=CLIPS[:1])[0] == "quit"
    assert lines(tmp_path) == []


# ---- nothing inside the repo -------------------------------------------------------------------------------
def test_refuses_a_directory_inside_the_repo():
    with pytest.raises(ValueError, match="repo"):
        rp.ensure_outside_repo(REPO / "personal")
    with pytest.raises(ValueError, match="repo"):
        rp.ensure_outside_repo(REPO)


def test_accepts_a_directory_outside_the_repo(tmp_path):
    rp.ensure_outside_repo(tmp_path)


def test_a_session_writes_only_under_its_directory(tmp_path):
    run(tmp_path, *TAKE, ENTER)
    assert not (REPO / "audio").exists() and not (REPO / "transcripts.jsonl").exists()
    assert (tmp_path / "audio" / "A1.wav").exists() and (tmp_path / "transcripts.jsonl").exists()
