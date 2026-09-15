"""Configuration loading: config/default.yaml overlaid by config/local.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "default.yaml"
LOCAL_PATH = ROOT / "config" / "local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Mode:
    name: str
    description: str = ""
    prompt: str = ""
    model: str | None = None                       # per-mode LLM override
    examples: list[dict[str, str]] = field(default_factory=list)  # [{in, out}]


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        data = yaml.safe_load(DEFAULT_PATH.read_text()) or {}
        for p in [LOCAL_PATH, path]:
            if p and Path(p).exists():
                data = _deep_merge(data, yaml.safe_load(Path(p).read_text()) or {})
        return cls(raw=data)

    # Convenience accessors -------------------------------------------------
    @property
    def audio(self) -> dict:
        return self.raw.get("audio", {})

    @property
    def stt(self) -> dict:
        return self.raw.get("stt", {})

    def stt_engine_config(self, name: str | None = None) -> tuple[str, dict]:
        name = name or self.stt.get("engine", "parakeet")
        return name, self.stt.get("engines", {}).get(name, {})

    @property
    def refine(self) -> dict:
        return self.raw.get("refine", {})

    @property
    def inject(self) -> dict:
        return self.raw.get("inject", {})

    @property
    def dictionary(self) -> list[str]:
        return list(self.raw.get("dictionary", []) or [])

    @property
    def mode_names(self) -> list[str]:
        return list(self.raw.get("modes", {}))

    def mode(self, name: str | None = None) -> Mode:
        name = name or self.raw.get("default_mode", "dictation")
        m = self.raw.get("modes", {}).get(name)
        if m is None:
            raise KeyError(f"Unknown mode '{name}'. Known: {self.mode_names}")
        return Mode(name=name, description=m.get("description", ""), prompt=m.get("prompt", ""),
                    model=m.get("model"), examples=list(m.get("examples") or []))
