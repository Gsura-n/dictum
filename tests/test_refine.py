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
    assert msgs[-1]["content"] == "<transcript>hello</transcript>"


def test_modes_have_examples_and_model_overrides():
    cfg = Config.load()
    assert cfg.mode("dictation").examples
    assert cfg.mode("dictation").model is None       # uses default
    assert cfg.mode("command").model                 # overrides


def test_refine_levels_stacks_modes():
    """`edit` runs `dictation_ft` first and refines its output; notes keep both stages."""
    from dictum.config import Config
    from dictum.refine.common import refine_levels
    from dictum.types import Refined, Transcript

    seen = []

    class FakeRefiner:
        def refine(self, transcript, mode):
            seen.append((mode.name, transcript.text))
            return Refined(text=f"<{mode.name}>{transcript.text}", backend="fake", mode=mode.name,
                           latency_s=1.0, source=transcript, notes=[f"{mode.name} note"])

    cfg = Config.load()
    tr = Transcript(text="raw speech", engine="eval", latency_s=0.0, audio_s=0.0)
    out = refine_levels(FakeRefiner(), cfg, tr, cfg.mode("edit"))

    assert [m for m, _ in seen] == ["dictation_ft", "edit"]
    assert seen[1][1] == "<dictation_ft>raw speech"      # stage two sees stage one's output
    assert out.source is tr                              # ...but the raw transcript is still the source
    assert out.latency_s == 2.0
    assert out.notes == ["dictation_ft: dictation_ft note", "edit note"]
