"""Hotkey names -> macOS virtual keycodes. Pure data, no platform imports.

Modifiers arrive as kCGEventFlagsChanged events. The device-dependent flag
bits (NX_DEVICE*KEYMASK) distinguish left from right, which the generic
kCGEventFlagMaskAlternate etc. do not.
"""
from __future__ import annotations

from dataclasses import dataclass

# name: (keycode, device-dependent flag mask)
MODIFIER_KEYS: dict[str, tuple[int, int]] = {
    "right_option": (61, 0x00000040),
    "left_option": (58, 0x00000020),
    "right_command": (54, 0x00000010),
    "left_command": (55, 0x00000008),
    "right_shift": (60, 0x00000004),
    "left_shift": (56, 0x00000002),
    "right_control": (62, 0x00002000),
    "left_control": (59, 0x00000001),
    "fn": (63, 0x00800000),
}

# Keys most keyboards do not use for anything, good for dedicated bindings.
FUNCTION_KEYS: dict[str, int] = {
    "f13": 105, "f14": 107, "f15": 113, "f16": 106, "f17": 64, "f18": 79, "f19": 80,
}


@dataclass(frozen=True)
class Binding:
    key: str
    keycode: int
    mask: int | None        # None for function keys (keyDown/keyUp, not flags)
    mode: str


def resolve_bindings(hold: dict[str, str], known_modes: list[str]) -> list[Binding]:
    out = []
    for key, mode in (hold or {}).items():
        key = key.lower()
        if mode != "auto" and mode not in known_modes:
            raise ValueError(f"hotkey '{key}' is bound to unknown mode '{mode}'. Known: {known_modes}")
        if key in MODIFIER_KEYS:
            code, mask = MODIFIER_KEYS[key]
            out.append(Binding(key, code, mask, mode))
        elif key in FUNCTION_KEYS:
            out.append(Binding(key, FUNCTION_KEYS[key], None, mode))
        else:
            raise ValueError(f"unsupported hotkey '{key}'. Use one of {sorted(MODIFIER_KEYS) + sorted(FUNCTION_KEYS)}")
    if not out:
        raise ValueError("no hotkeys configured under hotkeys.hold")
    return out
