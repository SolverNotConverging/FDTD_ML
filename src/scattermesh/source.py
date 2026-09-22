"""Analytic broadband plane wave with frequency-independent propagation angle."""

from dataclasses import dataclass

import numpy as np

from .constants import C0


@dataclass(frozen=True)
class PlaneWave:
    frequency: float
    width: float
    delay: float
    angle: float = 0.0  # radians, measured counterclockwise from +x
    origin: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if (
            len(self.origin) != 2
            or not np.isfinite(
                [self.frequency, self.width, self.delay, self.angle, *self.origin]
            ).all()
            or self.frequency <= 0
            or self.width <= 0
            or self.delay <= 0
        ):
            raise ValueError("Invalid plane wave")

    def retardation(self, x, y):
        return (
            (x - self.origin[0]) * np.cos(self.angle) + (y - self.origin[1]) * np.sin(self.angle)
        ) / C0

    def pulse(self, t):
        u = np.asarray(t) - self.delay
        return self.envelope(t) * np.cos(2 * np.pi * self.frequency * u)

    def envelope(self, t):
        return np.exp(-(((np.asarray(t) - self.delay) / self.width) ** 2))

    def electric(self, x, y, t):
        return self.pulse(t - self.retardation(x, y))
