"""Personal fine-tuning on the user's own corrections.

Starts from the published dictation adapter and continues training a few
hundred steps on the user's corrected dictations, mixed with replay rows from
the general training set so it does not forget general cleanup. Before
switching, it scores both adapters on a held-out slice of the user's own
corrections and only activates the personal adapter if it is at least as good.

Runs entirely on the machine. Typical: 200 corrections, ~10 minutes on an M-series Mac.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from . import history
from .adapters import PERSONAL, resolve
from .config import ROOT, Config
from .evals import wer

MIN_PAIRS = 50
SEED = 20260915


def _chat(system: str, src: str, tgt: str) -> dict:
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"<transcript>{src}</transcript>"},
                         {"role": "assistant", "content": tgt}]}


def _score(adapter: str | None, cfg: Config, pairs: list[dict]) -> float:
    """Mean WER of an adapter on held-out personal pairs (lower is better)."""
    from .dictionary import parse
    from .refine.mlx_refiner import MLXRefiner
    from .types import Transcript
    mcfg = dict(cfg.refine.get("mlx", {}))
    mcfg.update(adapter_path=adapter, adapter_fallback=None, use_personal=False)
    r = MLXRefiner(dictionary=cfg.dictionary, normalize=cfg.refine.get("normalize"), **mcfg)
    r.entries = parse(cfg.dictionary)
    mode = cfg.mode("dictation_ft")
    ws = [wer(r.refine(Transcript(text=p["src"], engine="p", latency_s=0, audio_s=0), mode).text, p["tgt"])
          for p in pairs]
    return sum(ws) / len(ws)


def run(console, iters: int | None = None, replay_ratio: float = 2.0, dry_run: bool = False) -> None:
    cfg = Config.load()
    pairs = history.training_pairs()
    if len(pairs) < MIN_PAIRS:
        raise SystemExit(f"{len(pairs)} corrected dictations; need at least {MIN_PAIRS}. "
                         "Correct outputs with `dictum correct` (or accept good ones with `dictum correct --ok`).")
    base_adapter = resolve(cfg.refine.get("mlx", {}).get("adapter_path"), use_personal=False) \
        or resolve(cfg.refine.get("mlx", {}).get("adapter_fallback"), use_personal=False)
    if not base_adapter:
        raise SystemExit("no base dictation adapter found locally or on the Hub; run setup first")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_hold = max(10, len(pairs) // 5)
    held, train = pairs[:n_hold], pairs[n_hold:]

    system = cfg.mode("dictation_ft").prompt.strip()
    rows = [_chat(system, p["src"], p["tgt"]) for p in train]
    replay_file = ROOT / "evals" / "data" / "finetune" / "dictation" / "train.jsonl"
    replay = []
    if replay_file.exists():
        lines = replay_file.read_text().splitlines()
        replay = [json.loads(x) for x in rng.sample(lines, min(len(lines), int(len(rows) * replay_ratio)))]
    all_rows = rows + replay
    rng.shuffle(all_rows)

    work = PERSONAL.parent / "personal-work"
    data = work / "data"
    shutil.rmtree(work, ignore_errors=True)
    data.mkdir(parents=True)
    n_valid = max(5, len(all_rows) // 10)
    (data / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows[:n_valid]) + "\n")
    (data / "train.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows[n_valid:]) + "\n")
    iters = iters or max(100, min(600, len(all_rows) // 4 * 2))     # ~2 epochs, capped
    console.print(f"personal pairs: {len(train)} train, {len(held)} held out; replay rows: {len(replay)}; iters: {iters}")
    if dry_run:
        console.print(f"[dim]dry run: data written to {data}[/]")
        return

    base_cfg = json.loads((Path(base_adapter) / "adapter_config.json").read_text())
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", cfg.refine.get("mlx", {}).get("model"),
           "--train", "--data", str(data), "--adapter-path", str(work / "adapter"),
           "--resume-adapter-file", str(Path(base_adapter) / "adapters.safetensors"),
           "--iters", str(iters), "--batch-size", "4", "--grad-checkpoint",
           "--clear-cache-threshold", "2GB", "--learning-rate", "5e-5",
           "--num-layers", str(base_cfg.get("num_layers", 16)), "--mask-prompt", "--max-seq-length", "512",
           "--steps-per-report", "20", "--steps-per-eval", str(max(50, iters // 2)), "--save-every", str(iters),
           "--seed", str(SEED)]
    console.print("[dim]training personal adapter...[/]")
    subprocess.run(cmd, check=True)

    with console.status("scoring base vs personal adapter on your held-out corrections..."):
        base_wer = _score(base_adapter, cfg, held)
        new_wer = _score(str(work / "adapter"), cfg, held)
    console.print(f"held-out word error: base {base_wer:.3f} -> personal {new_wer:.3f}")
    if new_wer <= base_wer:
        shutil.rmtree(PERSONAL, ignore_errors=True)
        shutil.move(str(work / "adapter"), str(PERSONAL))
        console.print(f"[green]personal adapter activated[/] ({PERSONAL}). `dictum personalize --reset` to undo.")
    else:
        console.print("[yellow]personal adapter was worse on your held-out corrections; keeping the base adapter[/]")
