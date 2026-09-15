"""Command-line entry points: dictum listen | refine | bench | devices | modes."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config

app = typer.Typer(add_completion=False, help="Local-first voice dictation.")
console = Console()

ENGINE_OPT = typer.Option(None, help="STT engine override: parakeet | whisper")
MODE_OPT = typer.Option(None, help="Refinement mode: dictation | command | academic")
REFINE_OPT = typer.Option(None, help="Refine backend override: passthrough | ollama")
INJECT_OPT = typer.Option(None, help="Inject backend override: stdout | clipboard")


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
    default_model = cfg.refine.get("ollama", {}).get("model", "")
    t = Table("mode", "model", "examples", "description")
    for name in cfg.mode_names:
        m = cfg.mode(name)
        t.add_row(name, m.model or f"{default_model} (default)", str(len(m.examples)), m.description)
    console.print(t)


@app.command()
def listen(
    engine: str = ENGINE_OPT, mode: str = MODE_OPT, refine: str = REFINE_OPT, inject: str = INJECT_OPT,
    device: int = typer.Option(None, help="Input device index (see `dictum devices`)"),
    save: Path = typer.Option(None, help="Also save each clip as WAV into this directory (for benchmarks)"),
):
    """Press Enter to start recording, Enter again to stop. Ctrl+C to quit."""
    from .pipeline import Pipeline

    cfg = Config.load()
    pipe = Pipeline.from_config(cfg, engine, refine, inject, device)
    console.print(f"[dim]mic: {pipe.capture.device_name} @ {pipe.capture.device_rate} Hz[/]")
    with console.status("Loading models (first run downloads STT weights)..."):
        load = pipe.warm_up(mode)
    console.print(f"[green]ready[/]  stt={pipe.stt.name} ({load['stt']:.1f}s)  "
                  f"refine={pipe.refiner.name} ({load['refine']:.1f}s)  "
                  f"inject={pipe.injector.name}  mode={cfg.mode(mode).name}")
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
            if clip.is_silent:
                console.print(f"[red]silent clip[/] (rms {clip.rms:.5f}, peak {clip.peak:.3f}). "
                              "Check mic permission for Terminal and the device shown above; "
                              "try --device N from `dictum devices`.")
                continue
            if save:
                import soundfile as sf
                n += 1
                sf.write(save / f"clip_{n:03d}.wav", clip.samples, clip.sample_rate)
            with console.status("transcribing..."):
                refined, t = pipe.process(clip, mode)
            if refined.text != refined.source.text:
                console.print(f"[dim]raw: {refined.source.text}[/]")
            console.print(f"[dim]{t} | level rms {clip.rms:.3f} peak {clip.peak:.2f}[/]")
    except (KeyboardInterrupt, EOFError):
        console.print("\nbye")


@app.command()
def refine(
    text: str = typer.Argument(..., help="Raw text to refine, as if it came from the STT stage"),
    mode: str = MODE_OPT,
    all_modes: bool = typer.Option(False, "--all", help="Run every mode and compare"),
):
    """Refine typed text without the microphone. Handy for iterating on prompts."""
    from .refine import create_refiner
    from .types import Transcript

    cfg = Config.load()
    refiner = create_refiner(cfg.refine, cfg.dictionary)
    tr = Transcript(text=text, engine="typed", latency_s=0.0, audio_s=0.0)
    names = cfg.mode_names if all_modes else [cfg.mode(mode).name]
    for name in names:
        m = cfg.mode(name)
        warm = refiner.warm_up(m)
        r = refiner.refine(tr, m)
        console.rule(f"{name}  [dim]{getattr(refiner, 'model_for', lambda _: '')(m)}  "
                     f"warm {warm:.1f}s  refine {r.latency_s:.2f}s[/]")
        console.print(r.text)


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
