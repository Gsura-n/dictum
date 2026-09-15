from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from ..types import AudioClip


def list_input_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                        "default_sr": d["default_samplerate"]})
    return out


class SoundDeviceCapture:
    """Push-to-talk style capture. Frames are appended from a callback thread."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Overflows etc. are worth knowing about but not fatal.
            print(f"[capture] {status}", flush=True)
        with self._lock:
            self._frames.append(indata[:, 0].copy())

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=self.channels, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> AudioClip:
        assert self._stream is not None, "start() was not called"
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            samples = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
        return AudioClip(samples=samples.astype(np.float32), sample_rate=self.sample_rate)
