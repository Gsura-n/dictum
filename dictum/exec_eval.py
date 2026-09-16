"""Execution-based scoring for command mode.

A generated command is correct if running it produces the same result as the
reference command: same (normalised) output and the same filesystem state
afterwards. This accepts every valid alternative (sed -n vs head | tail) that a
string or utility match wrongly rejects, and it is the approach used by
execution-based benchmarks such as InterCode-Bash.

Safety: commands run in a Docker container with no network, capped memory and
CPU, as throwaway copies of a fixture directory. Nothing touches the host.

A case only counts if the reference itself "works" in the fixture (exits 0 and
prints something or changes files). NL2Bash references often point at paths or
hosts that do not exist; those are reported as not executable, not as passes.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid

from .config import ROOT

SANDBOX_DIR = ROOT / "evals" / "sandbox"
IMAGE = "dictum-sandbox:1"


def _docker(*args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kw)


class Sandbox:
    def __init__(self, console=None):
        if shutil.which("docker") is None:
            raise SystemExit("execution scoring needs Docker. Install Docker Desktop and start it.")
        self.console = console
        if _docker("info").returncode != 0:
            raise SystemExit("Docker is installed but not running. Open Docker Desktop, wait for it to say "
                             "'Engine running', then try again.")
        if _docker("image", "inspect", IMAGE).returncode != 0:
            if console:
                console.print("[dim]building sandbox image (first time only, ~1 min)...[/]")
            r = _docker("build", "-t", IMAGE, str(SANDBOX_DIR))
            if r.returncode != 0:
                raise SystemExit(f"sandbox build failed:\n{r.stderr[-2000:]}")
        self._cache: dict[str, dict] = {}
        self.restarts = 0
        self._start()
        self.pristine = self.run("true")["fs"]

    def _start(self) -> None:
        self.name = f"dictum-sandbox-{uuid.uuid4().hex[:8]}"
        r = _docker("run", "-d", "--rm", "--name", self.name, "--network", "none",
                    "--memory", "1536m", "--cpus", "1", "--pids-limit", "256",
                    "--tmpfs", "/runs:size=1g", IMAGE)
        if r.returncode != 0:
            raise SystemExit(f"could not start sandbox: {r.stderr}")

    def _alive(self) -> bool:
        r = _docker("inspect", "-f", "{{.State.Running}}", self.name)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def _exec(self, cmd: str) -> dict | None:
        try:
            r = subprocess.run(["docker", "exec", "-i", self.name, "python3", "/opt/sandbox/runner.py"],
                               input=cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {"rc": -1, "timed_out": True, "stdout": "", "stderr": "host timeout", "fs": "timeout"}
        if r.stdout.strip():
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                pass
        return None   # infrastructure failure, not a command result

    def run(self, cmd: str) -> dict:
        """Runs cmd in a fresh fixture copy. If the sandbox itself breaks (a command
        exhausted memory or temp space and the container died), restart it and
        retry once, so one bad command cannot silently fail every later case."""
        if cmd in self._cache:
            return self._cache[cmd]
        res = self._exec(cmd)
        if res is None:
            if not self._alive() or self._exec("true") is None:
                _docker("rm", "-f", self.name)
                self._start()
                self.restarts += 1
            res = self._exec(cmd) or {"rc": -2, "timed_out": False, "stdout": "", "stderr": "sandbox error",
                                      "fs": "error", "infra_error": True}
        if not res.get("infra_error"):
            self._cache[cmd] = res
        return res

    def close(self) -> None:
        _docker("rm", "-f", self.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def normalize_stdout(s: str) -> list[str]:
    lines = []
    for line in s.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"(^|\s)\./", r"\1", line).rstrip("/")
        if line:
            lines.append(line)
    return sorted(lines)


def executable(ref: dict, pristine_fs: str) -> bool:
    return (not ref["timed_out"] and ref["rc"] == 0
            and (bool(ref["stdout"].strip()) or ref["fs"] != pristine_fs))


def equivalent(hyp: dict, ref: dict) -> bool:
    return (not hyp["timed_out"] and hyp["fs"] == ref["fs"]
            and normalize_stdout(hyp["stdout"]) == normalize_stdout(ref["stdout"]))


def score_rows(rows: list[dict], console=None) -> None:
    """Adds exec_executable / exec_match to command rows in place."""
    targets = [r for r in rows if r.get("mode") == "command" and r.get("reference")]
    if not targets:
        return
    with Sandbox(console) as sb:
        for i, r in enumerate(targets, 1):
            if console:
                console.print(f"[dim]executing {i}/{len(targets)}[/]" + " " * 20, end="\r")
            ref = sb.run(r["reference"])
            r["exec_executable"] = executable(ref, sb.pristine)
            if r["exec_executable"]:
                hyp = sb.run(r["output"]) if r["output"].strip() else {"timed_out": False, "fs": "", "stdout": ""}
                r["exec_match"] = equivalent(hyp, ref)
    if console:
        console.print(" " * 60, end="\r")
