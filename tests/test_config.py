from dictum.config import Config


def test_default_config_loads():
    cfg = Config.load()
    name, ecfg = cfg.stt_engine_config()
    assert name in ("parakeet", "whisper")
    assert "model" in ecfg
    assert cfg.mode().name == "dictation"
    assert cfg.mode("command").prompt


def test_registry_lists_engines():
    from dictum.stt import available_engines
    assert {"parakeet", "whisper"} <= set(available_engines())


def test_passthrough_refiner():
    from dictum.refine import Passthrough
    from dictum.types import Transcript
    cfg = Config.load()
    t = Transcript(text="hello", engine="x", latency_s=0.1, audio_s=1.0)
    r = Passthrough().refine(t, cfg.mode())
    assert r.text == "hello" and r.mode == "dictation"
