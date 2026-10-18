import yaml

from dictum.config import DEFAULT_PATH


def test_cli_and_hotkey_default_modes_match():
    """`dictum listen`/`refine` with no --mode must use the mode the hotkeys use."""
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    cli_default = cfg["default_mode"]
    app_default = cfg["apps"]["default"]
    right_command = cfg["hotkeys"]["hold"]["right_command"]
    assert cli_default == app_default == right_command, (
        f"default_mode={cli_default!r}, apps.default={app_default!r}, "
        f"hotkeys.hold.right_command={right_command!r}"
    )
