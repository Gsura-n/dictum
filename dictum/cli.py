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
INJECT_OPT = typer.Option("stdout", help="Inject backend: stdout | clipboard | macos")


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
def run(engine: str = ENGINE_OPT):
    """Always-on mode: hold the hotkey anywhere, talk, release to paste into the focused app."""
    from .app import DictumApp
    from . import permissions

    if permissions.input_monitoring() is False:
        permissions.input_monitoring(request=True)
        console.print("[red]Input Monitoring is not granted[/] for this terminal. macOS has opened the prompt; "
                      "enable it in System Settings > Privacy & Security > Input Monitoring, "
                      "then quit and reopen the terminal.")
        raise typer.Exit(1)
    cfg = Config.load()
    if cfg.inject.get("backend") == "macos" and permissions.accessibility() is False:
        permissions.accessibility(request=True)
        console.print("[red]Accessibility is not granted[/] for this terminal, so dictum cannot paste. "
                      "Enable it in System Settings > Privacy & Security > Accessibility, then reopen the terminal.")
        raise typer.Exit(1)
    DictumApp(cfg, engine, console).run()


@app.command()
def apps(delay: float = typer.Option(3.0, help="Seconds to wait so you can switch to the target app")):
    """Show the frontmost app's bundle id and which mode `auto` would pick."""
    import time
    from .context import frontmost_app, mode_for_app

    cfg = Config.load()
    console.print(f"switch to the app you want to check... ({delay:.0f}s)")
    time.sleep(delay)
    app = frontmost_app()
    if app is None:
        console.print("[red]could not read the frontmost app[/]")
        raise typer.Exit(1)
    mode = mode_for_app(app, cfg.raw.get("apps", {}), cfg.mode_names)
    console.print(f"app: [bold]{app.name}[/]  bundle id: [bold]{app.bundle_id}[/]  → auto mode: [green]{mode}[/]")


@app.command()
def check():
    """Check permissions, microphone and Ollama models."""
    from . import permissions

    cfg = Config.load()
    ok, bad, na = "[green]ok[/]", "[red]missing[/]", "[dim]n/a[/]"
    t = Table("check", "status", "detail", show_header=False)

    im, ax = permissions.input_monitoring(), permissions.accessibility()
    t.add_row("Input Monitoring", ok if im else (na if im is None else bad), "hotkey (dictum run)")
    t.add_row("Accessibility", ok if ax else (na if ax is None else bad), "paste into apps")

    try:
        from .capture import SoundDeviceCapture
        c = SoundDeviceCapture(**{k: v for k, v in cfg.audio.items() if k in ("sample_rate", "channels", "device")})
        t.add_row("Microphone", ok, f"{c.device_name} @ {c.device_rate} Hz")
    except Exception as e:
        t.add_row("Microphone", bad, str(e))

    name, ecfg = cfg.stt_engine_config()
    try:
        from .stt import create_engine
        create_engine(name, **ecfg)
        t.add_row("STT engine", ok, f"{name}: {ecfg.get('model')}")
    except Exception as e:
        t.add_row("STT engine", bad, str(e))

    if cfg.refine.get("backend") == "ollama":
        o = cfg.refine.get("ollama", {})
        needed = {o.get("model")} | {cfg.mode(m).model for m in cfg.mode_names if cfg.mode(m).model}
        try:
            import httpx
            tags = httpx.get(f"{o.get('host', 'http://localhost:11434')}/api/tags", timeout=3).json()
            have = {m["name"] for m in tags.get("models", [])}
            t.add_row("Ollama", ok, o.get("host", ""))
            for m in sorted(needed):
                present = m in have or f"{m}:latest" in have
                t.add_row(f"  model {m}", ok if present else bad, "" if present else f"ollama pull {m}")
        except Exception as e:
            t.add_row("Ollama", bad, f"not reachable ({type(e).__name__}); start Ollama")
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


@app.command("eval")
def eval_(
    suite: str = typer.Option("targeted", help="targeted | disflqa | nl2bash (see evals/DATASETS.md)"),
    split: str = typer.Option("dev", help="dev (tune against) | test (report only; do not read failures while tuning)"),
    mode: str = typer.Option(None, help="Only these modes, comma-separated"),
    models: str = typer.Option(None, help="Compare these Ollama models, comma-separated (default: mode's configured model)"),
    limit: int = typer.Option(None, help="Only the first N cases (quick smoke run)"),
    repeat: int = typer.Option(1, help="Runs per case (use 3 to see flakiness)"),
    failures: bool = typer.Option(None, "--failures/--no-failures", help="Print failing cases (default: on for dev, off for test)"),
    markdown: bool = typer.Option(False, help="Also print a markdown table for the README"),
    exec_: bool = typer.Option(False, "--exec", help="Command mode: also score by running commands in a Docker sandbox"),
    backend: str = typer.Option("ollama", help="ollama | mlx (mlx uses refine.mlx model + adapter from config)"),
    as_mode: str = typer.Option(None, help="Run every case under this mode instead (e.g. dictation_ft)"),
):
    """Score the refine stage: pass rate with 95% CI, latency, and suite-specific metrics."""
    from . import evals

    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be dev or test")
    cfg = Config.load()
    results = evals.run_eval(cfg, mode.split(",") if mode else None,
                             [m.strip() for m in models.split(",")] if models else None, repeat, console,
                             suite=suite, split=split, limit=limit, backend=backend, as_mode=as_mode)
    if exec_:
        from .exec_eval import score_rows
        score_rows(results["rows"], console)
    summary = evals.summarize(results)
    path = evals.save(results)
    _print_summary(summary, suite, split, markdown, path, results, failures)


def _print_summary(summary, suite, split, markdown, path, results, failures):
    from . import evals

    show = failures if failures is not None else (split == "dev")
    if show:
        for r in results["rows"]:
            if not r["passed"]:
                console.print(f"[red]✗[/] [bold]{r['mode']}/{r['case']}[/] [dim]({r['difficulty']}, {r['model']})[/]")
                console.print(f"   [dim]in: [/] {r['input']}")
                console.print(f"   [dim]ref:[/] {r['reference']}")
                console.print(f"   [dim]out:[/] {r['output']}")
                console.print(f"   [dim]why:[/] [yellow]{'; '.join(r['failures'])}[/]")
                if r.get("notes"):
                    console.print(f"   [dim]notes: {'; '.join(r['notes'])}[/]")
    elif split == "test":
        console.print("[dim]test split: failures hidden so they cannot leak into tuning (--failures to override)[/]")

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"
    t = Table("mode", "model", "n", "pass", "95% CI", "hard", "guard", "extra", "p50 s", "p95 s",
              title=f"Refine eval: {suite} / {split if suite != 'targeted' else 'all'}")
    for s in summary:
        extra = []
        if "mean_wer" in s:
            extra.append(f"wer {s['mean_wer']:.3f}")
        if "flag_f1" in s:
            extra.append(f"flag F1 {s['flag_f1']:.2f}")
        if "exec_pass" in s:
            extra.append(f"[bold]exec {s['exec_pass']:.0%}[/] ({s['exec_ci_low']:.0%} to {s['exec_ci_high']:.0%}, n={s['exec_n']})")
        t.add_row(s["mode"], s["model"], str(s["n"]), f"[bold]{pct(s['pass'])}[/]",
                  f"{s['ci_low']:.0%} to {s['ci_high']:.0%}", pct(s["hard"]), str(s["guard_trips"]),
                  ", ".join(extra), f"{s['p50_s']:.2f}", f"{s['p95_s']:.2f}")
    console.print(t)
    if markdown:
        console.print(evals.markdown_table(summary))
    console.print(f"[dim]full results: {path}[/]")


@app.command()
def rescore(
    results_file: Path = typer.Argument(..., help="A saved evals/results/*.json file"),
    exec_: bool = typer.Option(True, "--exec/--no-exec", help="Execution-based scoring for command rows"),
):
    """Re-apply scorers to saved outputs without calling the LLM again."""
    import json
    from . import evals

    results = json.loads(results_file.read_text())
    for r in results["rows"]:
        if r.get("suite") == "nl2bash" or (r.get("mode") == "command" and r["case"].startswith("nl2bash")):
            r.update(evals.command_score(r["output"], r["reference"]))
            r["passed"] = r["util_match"] and "\n" not in r["output"].strip()
    if exec_:
        from .exec_eval import score_rows
        score_rows(results["rows"], console)
    summary = evals.summarize(results)
    out = results_file.with_name(results_file.stem + "-rescored.json")
    out.write_text(json.dumps(dict(results, summary=summary), indent=2))
    _print_summary(summary, results.get("suite", ""), results.get("split", ""), False, out, results, False)


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
