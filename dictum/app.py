"""The always-on dictation app: hotkey -> record -> process -> paste.

Threads, and why:
- main:      Quartz run loop for the hotkey tap (macOS requires the main thread).
- recorder:  opens/closes the mic. Opening a Bluetooth input can take a few
             hundred ms, far too long to do inside the tap callback.
- processor: owns the models. MLX state is per thread, so weights are loaded
             and used on this one thread only. Clips queue up if you dictate
             faster than they process.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time

from rich.console import Console

from .config import Config
from .context import frontmost_app, mode_for_app
from .hotkey.keymap import resolve_bindings
from .pipeline import Pipeline

SOUNDS = {"start": "/System/Library/Sounds/Tink.aiff",
          "stop": "/System/Library/Sounds/Pop.aiff",
          "cancel": "/System/Library/Sounds/Basso.aiff"}


class DictumApp:
    def __init__(self, cfg: Config, engine: str | None = None, console: Console | None = None):
        self.cfg = cfg
        self.console = console or Console()
        hk = cfg.raw.get("hotkeys", {})
        self.bindings = resolve_bindings(hk.get("hold", {}), cfg.mode_names)
        self.sounds = bool(hk.get("sounds", True))
        self.min_hold_s = float(hk.get("min_hold_s", 0.35))
        self.pipe = Pipeline.from_config(cfg, engine)
        self._events: queue.Queue = queue.Queue()
        self._clips: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    # -- helpers -------------------------------------------------------------
    def _sound(self, name: str) -> None:
        if self.sounds:
            subprocess.Popen(["afplay", SOUNDS[name]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -- hotkey callbacks (tap thread: must be instant) -----------------------
    def _on_down(self, mode: str) -> None:
        self._events.put(("down", mode))

    def _on_up(self) -> None:
        self._events.put(("up", None))

    def _on_cancel(self) -> None:
        self._events.put(("cancel", None))

    # -- recorder thread -----------------------------------------------------
    def _recorder(self) -> None:
        mode, t_start = None, 0.0
        cap = self.pipe.capture
        while True:
            kind, arg = self._events.get()
            if kind == "down" and mode is None:
                # Resolve the mode at press time: the app in front now is the
                # one the text will be pasted into.
                app = frontmost_app()
                resolved = mode_for_app(app, self.cfg.raw.get("apps", {}), self.cfg.mode_names) if arg == "auto" else arg
                try:
                    cap.start()
                except Exception as e:
                    self.console.print(f"[red]mic error:[/] {e}")
                    continue
                mode, t_start = resolved, time.perf_counter()
                self._sound("start")
                where = f" in {app.name}" if app else ""
                self.console.print(f"[cyan]● recording[/] [dim]({mode}{where})[/]")
            elif kind in ("up", "cancel") and mode is not None:
                clip = cap.stop()
                held = time.perf_counter() - t_start
                m, mode = mode, None
                if kind == "cancel":
                    self._sound("cancel")
                    self.console.print("[yellow]cancelled[/] [dim](another key pressed)[/]")
                elif held < self.min_hold_s:
                    self.console.print("[dim]tap ignored (hold the key to dictate)[/]")
                elif clip.is_silent:
                    self._sound("cancel")
                    self.console.print(f"[red]silent clip[/] [dim]rms {clip.rms:.5f}; check the mic[/]")
                else:
                    self._sound("stop")
                    self._clips.put((clip, m))

    # -- processor thread ----------------------------------------------------
    def _processor(self) -> None:
        try:
            modes = {b.mode for b in self.bindings if b.mode != "auto"}
            if any(b.mode == "auto" for b in self.bindings):
                apps = self.cfg.raw.get("apps", {})
                modes |= {apps.get("default", "dictation")} | {r["mode"] for r in apps.get("rules", [])}
            modes = sorted(modes)
            load = self.pipe.warm_up(modes=modes)
            self.console.print(f"[dim]models loaded: stt {load['stt']:.1f}s, refine {load['refine']:.1f}s[/]")
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            clip, mode = self._clips.get()
            try:
                refined, t = self.pipe.process(clip, mode)
            except Exception as e:
                self.console.print(f"[red]error:[/] {type(e).__name__}: {e}")
                continue
            for note in refined.notes:
                self.console.print(f"[yellow]  {note}[/]")
            if refined.backend.startswith("failed"):
                self.console.print(f"[yellow]refine {refined.backend}; pasted raw transcript[/]")
            # Newline first: if you dictated into this terminal, the paste sits on the current line.
            self.console.print()
            if refined.text != refined.source.text:
                self.console.print(f"[dim]  raw: {refined.source.text}[/]")
            self.console.print(f"[green]✓[/] {refined.text}")
            self.console.print(f"[dim]  {t}[/]")

    # -- main ----------------------------------------------------------------
    def run(self) -> None:
        from .hotkey.macos_tap import HotkeyListener

        threading.Thread(target=self._processor, name="processor", daemon=True).start()
        with self.console.status("Loading models..."):
            self._ready.wait()
        if self._startup_error:
            raise self._startup_error
        threading.Thread(target=self._recorder, name="recorder", daemon=True).start()

        keys = ", ".join(f"hold [bold]{b.key.replace('_', ' ')}[/] → {b.mode}" for b in self.bindings)
        self.console.print(f"[green]dictum is listening.[/] {keys}. "
                           f"inject={self.pipe.injector.name}. Ctrl+C to quit.")
        listener = HotkeyListener(self.bindings, self._on_down, self._on_up, self._on_cancel)
        try:
            listener.run()
        except KeyboardInterrupt:
            listener.stop()
            self.console.print("\nbye")
