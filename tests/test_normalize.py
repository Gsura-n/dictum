import pytest

from dictum import normalize as n


def test_spoken_punctuation_and_breaks():
    segs = n.apply("dear team comma new paragraph the launch moved to Thursday period please update period new paragraph thanks comma Gauttam")
    assert [s for s, _ in segs] == ["dear team,", "the launch moved to Thursday. please update.", "thanks, Gauttam"]
    assert n.join(segs) == "dear team,\n\nthe launch moved to Thursday. please update.\n\nthanks, Gauttam"


def test_period_as_noun_kept():
    assert n.join(n.apply("the trial period ends Friday")) == "the trial period ends Friday"


def test_email_and_url():
    out = n.join(n.apply("you can reach me at g suraneni at gmail dot com or look at github dot com slash gsura dash n slash dictum"))
    assert "gsuraneni@gmail.com" in out and "github.com/gsura-n/dictum" in out
    assert n.join(n.apply("send it to j patel at outlook dot com")) == "send it to jpatel@outlook.com"


def test_no_false_addresses():
    assert n.join(n.apply("meet me at the office at noon")) == "meet me at the office at noon"


def test_repeats():
    assert n.collapse_repeats("the the report is is ready") == "the report is ready"
    assert n.collapse_repeats("so, it, it, it was great") == "so, it was great"
    assert n.collapse_repeats("I had had enough") == "I had had enough"


def test_refine_flow_with_fake_model():
    from dictum.config import Config
    from dictum.dictionary import parse
    from dictum.refine.common import refine_with
    from dictum.types import Transcript

    cfg = Config.load()
    mode = cfg.mode("dictation_ft")
    seen = []

    def fake(m, text):
        seen.append(text)
        return text[:1].upper() + text[1:]

    tr = Transcript(text="dear team comma new paragraph the the launch moved period new paragraph thanks comma gauttam",
                    engine="t", latency_s=0, audio_s=0)
    r = refine_with(fake, "fake", tr, mode, parse(cfg.dictionary), cfg.refine.get("normalize"))
    assert seen == ["dear team,", "the launch moved.", "thanks, Gauttam"]
    assert r.text == "Dear team,\n\nThe launch moved.\n\nThanks, Gauttam"


def test_scratch_that_inline_and_sentence_start():
    from dictum.normalize import scratch_that
    assert scratch_that("the budget is fifty thousand dollars scratch that we haven't finalized it") == "We haven't finalized it"
    assert scratch_that("Send it Monday. The budget is fifty. Scratch that. We haven't finalized it.") == \
        "Send it Monday. We haven't finalized it."
    assert scratch_that("Meet at noon. Order pizza scratch that order sushi.") == "Meet at noon. Order sushi."
    assert scratch_that("It was great, scratch that.") == ""


def test_scratch_that_real_phrases_untouched():
    from dictum.normalize import scratch_that
    for t in ["Don't scratch that, it will scar.", "I need to scratch that itch.", "Strike that off the list.",
              "We had to scratch that plan."]:
        assert scratch_that(t) == t, t


def test_scratch_that_in_apply_respects_paragraphs():
    from dictum import normalize as n
    segs = n.apply("hello team new paragraph the deadline is Monday scratch that it is Friday")
    assert n.join(segs) == "hello team\n\nIt is Friday"


def test_tech_dictionary_defaults():
    from dictum.config import Config
    from dictum.dictionary import apply, parse
    e = parse(Config.load().dictionary)
    assert apply("the use effect hook in type script on git hub", e) == "the useEffect hook in TypeScript on GitHub"
    assert apply("how did they react to the java script change", e) == "how did they react to the JavaScript change"


def test_chunk_keeps_corrections_with_their_sentence():
    from dictum.normalize import chunk
    t = ("I have a lot going on this week with several projects. So there is a deadline that I have to meet by Monday. "
         "No, not actually Monday. Thursday next week. Let's make a note of it as well.")
    parts = chunk(t, 12)
    joined = [p for p in parts if "Monday" in p]
    assert len(joined) == 1 and "No, not actually Monday." in joined[0]
    assert " ".join(parts) == t
    assert chunk("short text here", 60) == ["short text here"]


def test_command_mode_falls_back_for_long_input():
    import pytest
    try:
        import sounddevice  # noqa: F401  (pipeline imports the mic layer)
    except (ImportError, OSError):
        pytest.skip("sounddevice/PortAudio not available")
    from dictum.config import Config
    from dictum.pipeline import Pipeline
    from dictum.types import Refined, Transcript

    class FakeRefiner:
        name = "fake"
        def refine(self, t, mode):
            return Refined(text=mode.name, backend="fake", mode=mode.name, latency_s=0, source=t)

    cfg = Config.load()
    p = Pipeline(capture=None, stt=None, refiner=FakeRefiner(), injector=None, cfg=cfg)
    long_t = Transcript(text=" ".join(["word"] * 50), engine="x", latency_s=0, audio_s=0)
    short_t = Transcript(text="list all python files here", engine="x", latency_s=0, audio_s=0)
    r = p.refine_text(long_t, "command")
    assert r.mode == "dictation_ft" and "too long" in r.notes[0]
    assert p.refine_text(short_t, "command").mode == "command"


def test_chunk_splits_unpunctuated_monologue_at_clauses():
    from dictum.normalize import chunk
    t = ("So there might be multiple things that we are looking into and there are also multiple projects that I'm "
         "working on I'm also trying to understand how I be able to refine this and make it better and better and "
         "better and I'm also trying to understand how will I be able to train it in such a way that it works "
         "perfectly it works as expected and also the most important thing here I think is that how would we able "
         "to transfer all this training and refinements in such a way that when this is adopted by a different "
         "system let's say if I move the same project to a different device I don't have to do all the trainings")
    parts = chunk(t, 60)
    assert len(parts) >= 2
    assert all(len(p.split()) <= 78 for p in parts)
    assert " ".join(parts).split() == t.split()           # no words lost or added
    assert all(len(p.split()) >= 15 for p in parts[:-1])  # no tiny fragments


@pytest.mark.parametrize("raw,want", [
    # false start: the speaker begins the phrase again, more completely
    ("how would be how I would be able to do this", "how I would be able to do this"),
    ("and I want to I want to try this again", "and I want to try this again"),
    ("I think I think we should ship it", "I think we should ship it"),
    ("give me the red one the red one please", "give me the red one please"),
    # left alone: ordinary English that happens to repeat a word
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("the more you practice the more you improve", "the more you practice the more you improve"),
    ("in the end the end justifies the means", "in the end the end justifies the means"),
    ("at the end of the day the day is short", "at the end of the day the day is short"),
    ("it's not just a car it's a spaceship", "it's not just a car it's a spaceship"),
    # a substitution repair is the model's job, not a rule's
    ("I went to the store I went to the market", "I went to the store I went to the market"),
    # never across a sentence end
    ("She said that. That was the plan.", "She said that. That was the plan."),
])
def test_collapse_restarts(raw, want):
    assert n.collapse_restarts(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("we can make this better and better and better and better", "we can make this better and better"),
    ("it goes on and on and on", "it goes on and on"),
    ("again and again and again and again", "again and again"),
    # two copies is the idiom, not a stutter
    ("it goes on and on", "it goes on and on"),
    ("tea or coffee or juice", "tea or coffee or juice"),
    ("salt and pepper and olive oil", "salt and pepper and olive oil"),
])
def test_reduce_phrase_repeats(raw, want):
    assert n.reduce_phrase_repeats(raw) == want
