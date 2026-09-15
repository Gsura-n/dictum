"""Command-line entry points: dictum listen | bench | devices | modes."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config

app = typer.Typer(add_completion=False, help="Local-first voice dictation.")
console = Console()


@app.command()
def devices():
    """List audio input devices."""
    from .capture import list_input_devices
    t = Table("index", "name", "channels", "default sr")
    for d in list_input_devices():
        t.add_row(str(d["index"]), d["name"], str(d["channels"]), str(int(d["default_sr"])))
    console.print(t)


@app.command()
def modes():
    """List refinement modes from config."""
    cfg = Config.load()
    t = Table("mode", "description")
    for name, m in cfg.raw.get("modes", {}).items():
        t.add_row(name, m.get("description", ""))
    console.print(t)


@app.command()
def listen(
    engine: str = typer.Option(None, help="STT engine override: parakeet | whisper"),
    mode: str = typer.Option(None, help="Refinement mode: dictation | command | academic"),
    save: Path = typer.Option(None, help="Also save each clip as WAV into this directory (for benchmarks)"),
):
    """Press Enter to start recording, Enter again to stop. Ctrl+C to quit."""
    from .pipeline import Pipeline

    cfg = Config.load()
    pipe = Pipeline.from_config(cfg, engine)
    with console.status(f"Loading {pipe.stt.name} model (first run downloads weights)..."):
        load_s = pipe.warm_up()
    console.print(f"[green]ready[/] ({pipe.stt.name}, loaded in {load_s:.1f}s). "
                  f"refine={pipe.refiner.name} inject={pipe.injector.name} mode={cfg.mode(mode).name}")
    if save:
        save.mkdir(parents=True, exist_ok=True)

    n = 0
    try:
        while True:
            input("\n[Enter] to start recording...")
            pipe.capture.start()
            input("recording... [Enter] to stop")
            clip = pipe.capture.stop()
            if clip.duration_s < 0.3:
                console.print("[yellow]too short, skipped[/]")
                continue
            if save:
                import soundfile as sf
                n += 1
                sf.write(save / f"clip_{n:03d}.wav", clip.samples, clip.sample_rate)
            with console.status("transcribing..."):
                refined, t = pipe.process(clip, mode)
            console.print(f"[dim]audio {t.audio_s:.1f}s | stt {t.stt_s:.2f}s (rtf {t.stt_s / t.audio_s:.2f}) "
                          f"| refine {t.refine_s:.2f}s | total {t.total_s:.2f}s[/]")
    except (KeyboardInterrupt, EOFError):
        console.print("\nbye")


@app.command()
def bench(
    samples: Path = typer.Option(Path("benchmarks/samples"), help="Directory of .wav files (+ optional .txt references)"),
    engines: str = typer.Option("parakeet,whisper", help="Comma-separated engines to compare"),
    out: Path = typer.Option(Path("benchmarks/results"), help="Where to write results"),
):
    """Compare STT engines on latency and (if references exist) word error rate."""
    from benchmarks.bench_stt import run_benchmark
    run_benchmark(samples, [e.strip() for e in engines.split(",")], out, console)


if __name__ == "__main__":
    app()
