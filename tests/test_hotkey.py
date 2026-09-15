import pytest

from dictum.config import Config
from dictum.hotkey.keymap import resolve_bindings


def test_default_hotkeys_resolve():
    cfg = Config.load()
    b = resolve_bindings(cfg.raw["hotkeys"]["hold"], cfg.mode_names)
    keys = {x.key: x for x in b}
    assert keys["right_option"].keycode == 61 and keys["right_option"].mode == "auto"
    assert keys["right_option"].mask == 0x40


def test_function_key_has_no_mask():
    (b,) = resolve_bindings({"f13": "dictation"}, ["dictation"])
    assert b.mask is None and b.keycode == 105


def test_bad_bindings_rejected():
    with pytest.raises(ValueError):
        resolve_bindings({"space": "dictation"}, ["dictation"])
    with pytest.raises(ValueError):
        resolve_bindings({"right_option": "nope"}, ["dictation"])
