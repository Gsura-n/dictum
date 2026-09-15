from .base import Injector
from .stdout import StdoutInjector, ClipboardInjector

__all__ = ["Injector", "StdoutInjector", "ClipboardInjector", "create_injector"]


def create_injector(cfg: dict) -> Injector:
    backend = cfg.get("backend", "stdout")
    if backend == "stdout":
        return StdoutInjector()
    if backend == "clipboard":
        return ClipboardInjector()
    if backend == "macos":
        from .macos import MacOSInjector
        return MacOSInjector()
    raise KeyError(f"Unknown inject backend '{backend}'")
