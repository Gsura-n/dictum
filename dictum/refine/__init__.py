from .base import Refiner
from .passthrough import Passthrough

__all__ = ["Refiner", "Passthrough", "create_refiner"]


def create_refiner(cfg: dict, dictionary: list[str], backend: str | None = None) -> Refiner:
    backend = backend or cfg.get("backend", "passthrough")
    if backend == "passthrough":
        return Passthrough()
    if backend == "ollama":
        from .ollama_refiner import OllamaRefiner
        return OllamaRefiner(dictionary=dictionary, normalize=cfg.get("normalize"), **cfg.get("ollama", {}))
    if backend == "mlx":
        from .mlx_refiner import MLXRefiner
        return MLXRefiner(dictionary=dictionary, normalize=cfg.get("normalize"), **cfg.get("mlx", {}))
    raise KeyError(f"Unknown refine backend '{backend}'")
