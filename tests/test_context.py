import pytest

from dictum.config import Config
from dictum.context import AppInfo, mode_for_app
from dictum.hotkey.keymap import resolve_bindings


def test_app_rules_pick_mode_by_bundle_or_name():
    cfg = Config.load()
    apps = cfg.raw["apps"]
    assert mode_for_app(AppInfo("com.apple.Terminal", "Terminal"), apps, cfg.mode_names) == "command"
    assert mode_for_app(AppInfo("x.y", "iTerm2"), apps, cfg.mode_names) == "dictation_ft"
    assert mode_for_app(AppInfo("com.microsoft.Word", "Microsoft Word"), apps, cfg.mode_names) == "academic"
    assert mode_for_app(AppInfo("com.tinyspeck.slackmacgap", "Slack"), apps, cfg.mode_names) == "dictation_ft"
    assert mode_for_app(None, apps, cfg.mode_names) == "dictation_ft"


def test_match_by_name():
    rules = {"default": "dictation", "rules": [{"mode": "command", "match": ["Warp"]}]}
    assert mode_for_app(AppInfo("dev.warp.Warp-Stable", "Warp"), rules, ["dictation", "command"]) == "command"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        mode_for_app(AppInfo("a", "A"), {"rules": [{"mode": "nope", "match": ["a"]}]}, ["dictation"])


def test_auto_binding_allowed():
    cfg = Config.load()
    b = resolve_bindings(cfg.raw["hotkeys"]["hold"], cfg.mode_names)
    assert {x.mode for x in b} == {"auto", "dictation_ft", "edit"}
