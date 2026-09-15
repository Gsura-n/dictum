"""What is the user looking at? Phase 4 starts with the frontmost app;
focused-field text and selection come next (voice editing)."""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class AppInfo:
    bundle_id: str
    name: str


def frontmost_app() -> AppInfo | None:
    if sys.platform != "darwin":
        return None
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        return AppInfo(bundle_id=str(app.bundleIdentifier() or ""), name=str(app.localizedName() or ""))
    except Exception:
        return None


def mode_for_app(app: AppInfo | None, apps_cfg: dict, known_modes: list[str]) -> str:
    default = (apps_cfg or {}).get("default", "dictation")
    if app is not None:
        for rule in (apps_cfg or {}).get("rules", []) or []:
            matches = {m.lower() for m in rule.get("match", [])}
            if app.bundle_id.lower() in matches or app.name.lower() in matches:
                mode = rule["mode"]
                if mode not in known_modes:
                    raise ValueError(f"apps rule uses unknown mode '{mode}'")
                return mode
    return default
