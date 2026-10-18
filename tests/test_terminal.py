import random
import unicodedata

import pytest

from dictum.config import Config
from dictum.context import AppInfo
from dictum.terminal import flatten_reason, terminal_safe

TERM = AppInfo("com.apple.Terminal", "Terminal")
WORD = AppInfo("com.microsoft.Word", "Word")


@pytest.mark.parametrize("text, expected", [
    ("ls -la\n", "ls -la"),
    ("ls -la\r\n", "ls -la"),
    ("a\n\nb", "a b"),
    ("a\r\nb\rc", "a b c"),
    ("a\tb", "a b"),
    ("\x1b[31mred\x1b[0m", "[31mred[0m"),         # ESC dropped, the rest stays visible
    ("a\x00b\x7fc", "abc"),
    ("a b c\x85d\x0be\x0cf", "a b c d e f"),
    ("one\n\n\ntwo\n  three  \n", "one two three"),
    ("", ""),
    ("\n\n", ""),
    ("echo hi", "echo hi"),
    ("Hello, world. It's 5 o'clock; ok?", "Hello, world. It's 5 o'clock; ok?"),
    ("café — naïve 日本語", "café — naïve 日本語"),
])
def test_terminal_safe_cases(text, expected):
    assert terminal_safe(text) == expected


def test_terminal_safe_never_drops_words_and_is_idempotent():
    rng = random.Random(1234)
    pieces = ["alpha", "beta", "x1", "ls", "-la", "\n", "\r\n", "\t", "  ", "\x1b", " ", "\n\n"]
    for _ in range(500):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        out = terminal_safe(text)
        assert not any(unicodedata.category(c) == "Cc" for c in out)
        assert out == out.rstrip()
        assert terminal_safe(out) == out
        # no word is dropped: same words as the input once non-whitespace controls are removed
        kept = "".join(c for c in text if unicodedata.category(c) != "Cc" or c.isspace())
        assert out.split() == kept.split()


def test_flatten_reason_uses_apps_terminals():
    cfg = Config.load().raw["apps"]
    assert flatten_reason(cfg, TERM) == "terminal"
    assert flatten_reason(cfg, WORD) is None
    assert flatten_reason(cfg, WORD, TERM) == "terminal"        # press OR paste
    assert flatten_reason(cfg, TERM, WORD) == "terminal"


def test_flatten_reason_unknown_app_fails_closed():
    cfg = Config.load().raw["apps"]
    assert flatten_reason(cfg, None) == "unknown"
    assert flatten_reason(cfg, WORD, None) == "unknown"
    assert flatten_reason(cfg, None, TERM) == "terminal"        # a real hit wins over unknown
    assert flatten_reason(cfg, None, fail_closed=False) is None


def test_terminal_list_is_the_command_rule_list():
    """One list: the command rule must be the same ids as apps.terminals."""
    import yaml
    from dictum.config import DEFAULT_PATH
    apps = yaml.safe_load(DEFAULT_PATH.read_text())["apps"]
    (command,) = [r for r in apps["rules"] if r["mode"] == "command"]
    assert apps["terminals"] and command["match"] == apps["terminals"]
