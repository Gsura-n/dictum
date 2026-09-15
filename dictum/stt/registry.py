"""Name -> engine factory. Adding an engine = one file + one line here."""
from __future__ import annotations

from importlib import import_module

from .base import STTEngine

_REGISTRY: dict[str, tuple[str, str]] = {
    # name: (module, class)
    "parakeet": ("dictum.stt.parakeet_mlx", "ParakeetMLX"),
    "whisper": ("dictum.stt.whisper_mlx", "WhisperMLX"),
}


def available_engines() -> list[str]:
    return list(_REGISTRY)


def create_engine(name: str, **kwargs) -> STTEngine:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown STT engine '{name}'. Available: {available_engines()}")
    module, cls = _REGISTRY[name]
    try:
        mod = import_module(module)
    except ImportError as e:
        raise ImportError(
            f"Engine '{name}' needs an optional dependency. Install with: "
            f"pip install 'dictum[{name}]'  (underlying error: {e})"
        ) from e
    return getattr(mod, cls)(**kwargs)
