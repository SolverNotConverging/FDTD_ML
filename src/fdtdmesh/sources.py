"""Waveforms evaluated in one vectorized setup call, before uploading to CUDA."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waveform:
    kind: str = "gaussian"
    amplitude: float = 1.0
    frequency: float | None = None
    delay: float | None = None
    width: float | None = None
    phase: float = 0.0

    def sample(self, times, f_max):
        f = f_max / 2 if self.frequency is None else self.frequency
        width = 1 / f_max if self.width is None else self.width
        delay = 4 * width if self.delay is None else self.delay
        if (
            not np.isfinite([f, width, delay, self.amplitude, self.phase]).all()
            or f <= 0
            or width <= 0
            or delay < 0
        ):
            raise ValueError("Invalid waveform parameters")
        gaussian = np.exp(-(((times - delay) / width) ** 2))
        carrier = np.sin(2 * np.pi * f * (times - delay) + self.phase)
        if self.kind == "gaussian":
            return self.amplitude * gaussian
        if self.kind == "gaussian_sine":
            return self.amplitude * gaussian * carrier
        if self.kind in ("sine", "cw"):
            return self.amplitude * np.sin(2 * np.pi * f * times + self.phase)
        raise ValueError("waveform must be gaussian, gaussian_sine, sine, or cw")
