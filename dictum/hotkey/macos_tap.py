"""Global hold-to-talk listener built on a Quartz event tap.

Why a raw CGEventTap instead of pynput: pynput's macOS backend calls text
input APIs off the main thread, which crashes on recent macOS. A listen-only
tap on the main run loop is a few dozen lines, has no such problem, and lets
us tell left and right modifiers apart.

The callback must return quickly or macOS disables the tap, so it only
forwards events to the controller via callbacks that enqueue work.

Behaviour:
- press bound key  -> on_down(mode)
- release it       -> on_up()
- any other key pressed while holding -> on_cancel()  (it was a shortcut,
  e.g. Option+Shift+K, not a dictation)

Requires Input Monitoring permission for the terminal running dictum.
"""
from __future__ import annotations

from typing import Callable

import Quartz as Q

from .keymap import Binding


class HotkeyListener:
    def __init__(self, bindings: list[Binding], on_down: Callable[[str], None],
                 on_up: Callable[[], None], on_cancel: Callable[[], None]):
        self._mods = {b.keycode: b for b in bindings if b.mask is not None}
        self._fkeys = {b.keycode: b for b in bindings if b.mask is None}
        self.on_down, self.on_up, self.on_cancel = on_down, on_up, on_cancel
        self._active: Binding | None = None
        self._tap = None
        self._stop = False

    # -- event handling ------------------------------------------------------
    def _press(self, b: Binding) -> None:
        if self._active is None:
            self._active = b
            self.on_down(b.mode)

    def _release(self, b: Binding) -> None:
        if self._active is not None and self._active.keycode == b.keycode:
            self._active = None
            self.on_up()

    def _cancel(self) -> None:
        if self._active is not None:
            self._active = None
            self.on_cancel()

    def _callback(self, proxy, etype, event, refcon):
        if etype in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
            Q.CGEventTapEnable(self._tap, True)
            return event

        code = Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode)

        if etype == Q.kCGEventFlagsChanged:
            b = self._mods.get(code)
            if b is not None:
                pressed = bool(Q.CGEventGetFlags(event) & b.mask)
                self._press(b) if pressed else self._release(b)
            elif self._active is not None:
                self._cancel()          # another modifier joined in
        elif etype == Q.kCGEventKeyDown:
            b = self._fkeys.get(code)
            if b is not None:
                if not Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventAutorepeat):
                    self._press(b)
            elif self._active is not None:
                self._cancel()
        elif etype == Q.kCGEventKeyUp:
            b = self._fkeys.get(code)
            if b is not None:
                self._release(b)
        return event

    # -- run loop ------------------------------------------------------------
    def run(self) -> None:
        """Blocks on the current (main) thread until stop() or Ctrl+C."""
        mask = (Q.CGEventMaskBit(Q.kCGEventFlagsChanged)
                | Q.CGEventMaskBit(Q.kCGEventKeyDown)
                | Q.CGEventMaskBit(Q.kCGEventKeyUp))
        self._tap = Q.CGEventTapCreate(
            Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap, Q.kCGEventTapOptionListenOnly,
            mask, self._callback, None,
        )
        if self._tap is None:
            raise PermissionError(
                "Could not create the keyboard event tap. Grant Input Monitoring to your terminal: "
                "System Settings > Privacy & Security > Input Monitoring, then restart the terminal."
            )
        source = Q.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Q.CFRunLoopAddSource(Q.CFRunLoopGetCurrent(), source, Q.kCFRunLoopCommonModes)
        Q.CGEventTapEnable(self._tap, True)
        # Short slices instead of CFRunLoopRun() so Python can deliver Ctrl+C.
        while not self._stop:
            Q.CFRunLoopRunInMode(Q.kCFRunLoopDefaultMode, 0.25, False)

    def stop(self) -> None:
        self._stop = True
