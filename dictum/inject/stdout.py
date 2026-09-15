from __future__ import annotations

import subprocess
import sys


class StdoutInjector:
    name = "stdout"

    def inject(self, text: str) -> None:
        print(text, flush=True)


class ClipboardInjector:
    """Copies to the macOS clipboard via pbcopy. No permissions required."""

    name = "clipboard"

    def inject(self, text: str) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("clipboard injector currently supports macOS only")
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
