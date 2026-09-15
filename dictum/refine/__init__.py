from .base import Refiner
from .passthrough import Passthrough

__all__ = ["Refiner", "Passthrough", "create_refiner"]


def create_refiner(cfg: dict, dictionary: list[str]) -> Refiner:
    backend = cfg.get("backend", "passthrough")
    if backend == "passthrough":
        return Passthrough()
    if backend == "ollama":
        from .ollama_refiner import OllamaRefiner
        return OllamaRefiner(dictionary=dictionary, **cfg.get("ollama", {}))
    raise KeyError(f"Unknown refine backend '{backend}'")
