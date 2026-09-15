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


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    import soxr  # high quality, fast; a hard dependency because mic rates vary
    return soxr.resample(samples, src_sr, dst_sr).astype(np.float32)


class SoundDeviceCapture:
    """Push-to-talk style capture.

    Opens the device at its own native sample rate (Bluetooth headsets in
    particular return silence or fail when asked for a rate they don't run at)
    and resamples to the pipeline rate on stop().
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1, device=None):
        self.sample_rate = sample_rate          # what the pipeline wants
        self.channels = channels
        self.device = device
        try:
            # kind="input" resolves the system default input; sd.default.device
            # can be -1 ("unset") which query_devices rejects.
            info = sd.query_devices(device) if device is not None else sd.query_devices(kind="input")
        except sd.PortAudioError as e:
            raise RuntimeError(
                "No usable input device. Connect a mic (AirPods must be connected, not just paired) "
                "and check System Settings > Sound > Input."
            ) from e
        self.device_name = info["name"]
        self.device_rate = int(info["default_samplerate"])   # what the mic runs at
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[capture] {status}", flush=True)
        with self._lock:
            self._frames.append(indata[:, 0].copy())

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.device_rate, channels=self.channels, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> AudioClip:
        assert self._stream is not None, "start() was not called"
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            raw = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
        samples = resample(raw.astype(np.float32), self.device_rate, self.sample_rate)
        return AudioClip(samples=samples, sample_rate=self.sample_rate)
