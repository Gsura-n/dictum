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
