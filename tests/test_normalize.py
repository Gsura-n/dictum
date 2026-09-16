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
