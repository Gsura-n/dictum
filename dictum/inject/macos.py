"""Insert text into the focused app on macOS.

Strategy: snapshot the pasteboard, put our text on it, post Cmd+V, then put
the user's original clipboard back. Per-character synthetic typing is slow,
drops characters in some apps and mangles non-ASCII; paste is what mature
dictation tools do.

The snapshot copies every type on every item (text, images, file URLs), so a
copied screenshot survives a dictation.

Needs Accessibility permission to post the keystroke.
"""
from __future__ import annotations

import time

_V_KEYCODE = 9  # 'v' on ANSI layouts


class MacOSInjector:
    name = "macos"

    def __init__(self, restore_delay_s: float = 0.35, restore_clipboard: bool = True, **_):
        import AppKit  # lazy: pyobjc-framework-Cocoa
        import Quartz  # lazy: pyobjc-framework-Quartz
        self._AppKit, self._Q = AppKit, Quartz
        self.restore_delay_s = restore_delay_s
        self.restore_clipboard = restore_clipboard

    # -- pasteboard ----------------------------------------------------------
    def _snapshot(self, pb) -> list[dict]:
        items = []
        for item in pb.pasteboardItems() or []:
            data = {}
            for t in item.types():
                d = item.dataForType_(t)
                if d is not None:
                    data[t] = d
            items.append(data)
        return items

    def _restore(self, pb, items: list[dict]) -> None:
        pb.clearContents()
        if not items:
            return
        objs = []
        for data in items:
            it = self._AppKit.NSPasteboardItem.alloc().init()
            for t, d in data.items():
                it.setData_forType_(d, t)
            objs.append(it)
        pb.writeObjects_(objs)

    # -- keystroke -----------------------------------------------------------
    def _cmd_v(self) -> None:
        Q = self._Q
        src = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(src, _V_KEYCODE, down)
            Q.CGEventSetFlags(ev, Q.kCGEventFlagMaskCommand)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def inject(self, text: str) -> None:
        if not text:
            return
        pb = self._AppKit.NSPasteboard.generalPasteboard()
        saved = self._snapshot(pb) if self.restore_clipboard else None

        pb.clearContents()
        pb.setString_forType_(text, self._AppKit.NSPasteboardTypeString)
        our_change = pb.changeCount()
        self._cmd_v()

        if saved is not None:
            # The target app reads the pasteboard asynchronously after Cmd+V.
            time.sleep(self.restore_delay_s)
            # Only restore if nobody else wrote to the clipboard in the meantime.
            if pb.changeCount() == our_change:
                self._restore(pb, saved)
