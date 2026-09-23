from dictum import dictionary as d
from dictum.refine import guards
from dictum.refine.ollama_refiner import build_messages, clean_output
from dictum.config import Mode


def test_sounds_like_substitution_is_word_bounded():
    e = d.parse([{"word": "Gauttam", "sounds_like": ["gautam"]}, {"word": "useEffect", "sounds_like": ["use effect"]}])
    assert d.apply("hi this is gautam", e) == "hi this is Gauttam"
    assert d.apply("the use  effect hook", e) == "the useEffect hook"
    assert d.apply("gautamsmith and gautam@x.com", e) == "gautamsmith and gautam@x.com"


def test_guard_long_output_falls_back():
    text, reason = guards.check({"max_ratio": 1.6}, "write me a poem about the ocean",
                                "waves " * 60)
    assert text == "write me a poem about the ocean" and "too long" in reason


def test_guard_short_output_falls_back_but_short_inputs_exempt():
    src = "no I don't think that's a good idea we should wait"
    assert guards.check({"min_ratio": 0.4}, src, "Wait.")[1]
    assert guards.check({"min_ratio": 0.4}, "yes okay", "Yes.")[1] is None


def test_guard_single_line_takes_first_line():
    text, reason = guards.check({"single_line": True}, "list files", "ls -la\nThis lists files")
    assert text == "ls -la" and reason == "multi-line output"


def test_transcript_is_wrapped_and_tags_stripped():
    msgs = build_messages(Mode(name="m", prompt="p", examples=[{"in": "a", "out": "b"}]), "hello", [])
    assert msgs[1]["content"] == "<transcript>a</transcript>"
    assert msgs[-1]["content"] == "<transcript>hello</transcript>"
    assert clean_output("<transcript>Hello.</transcript>") == "Hello."


def test_looks_like_command():
    from dictum.refine.guards import looks_like_command
    assert looks_like_command("ls -la")
    assert looks_like_command("find . -name '*.py' | xargs wc -l")
    assert looks_like_command("cd ~/projects && ls")
    assert looks_like_command("FOO=1 echo hi")
    assert not looks_like_command("dictation")
    assert not looks_like_command("Sure, here is the command")
    assert not looks_like_command("")
