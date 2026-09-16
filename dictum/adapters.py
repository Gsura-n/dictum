"""Resolve LoRA adapter locations.

adapter_path may be:
  - a local folder (adapters/dictation-v2)
  - "hf:<user>/<repo>" to download a published adapter from the Hugging Face Hub
    (cached by huggingface_hub, so it downloads once)
Resolution order for the personal adapter: ~/.dictum/adapters/personal wins over
the configured one when personalization is enabled and that folder exists.
"""
from __future__ import annotations

from pathlib import Path

HOME = Path.home() / ".dictum"
PERSONAL = HOME / "adapters" / "personal"


def resolve(adapter_path: str | None, use_personal: bool = True) -> str | None:
    if use_personal and (PERSONAL / "adapters.safetensors").exists():
        return str(PERSONAL)
    if not adapter_path:
        return None
    if adapter_path.startswith("hf:"):
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=adapter_path[3:],
                                     allow_patterns=["adapters.safetensors", "adapter_config.json"])
        except Exception as e:   # offline, not published yet, or huggingface_hub missing
            print(f"[adapters] could not fetch {adapter_path}: {type(e).__name__}; using base model", flush=True)
            return None
    p = Path(adapter_path)
    if not p.is_absolute():
        from .config import ROOT
        p = ROOT / p
    return str(p) if (p / "adapters.safetensors").exists() else None
