from dictum.config import Config, Mode
from dictum.refine.ollama_refiner import build_messages, clean_output


def test_clean_output_strips_think_fences_quotes():
    assert clean_output("<think>reasoning</think>\nhello") == "hello"
    assert clean_output("```bash\nls -la\n```") == "ls -la"
    assert clean_output('"quoted text"') == "quoted text"
    assert clean_output("plain") == "plain"


def test_build_messages_includes_examples_and_dictionary():
    mode = Mode(name="t", prompt="Do X.", examples=[{"in": "a", "out": "b"}])
    msgs = build_messages(mode, "hello", ["Gauttam"])
    assert msgs[0]["role"] == "system" and "Gauttam" in msgs[0]["content"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "hello"


def test_modes_have_examples_and_model_overrides():
    cfg = Config.load()
    assert cfg.mode("dictation").examples
    assert cfg.mode("dictation").model is None       # uses default
    assert cfg.mode("command").model                 # overrides
