"""The single POC source: a finite Gaussian-modulated sinusoid, in SI units."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianPulse:
    fmin: float
    fmax: float
    edge_level_db: float = -6.0
    cutoff: float = 5.0

    def __post_init__(self):
        if (
            not np.isfinite([self.fmin, self.fmax, self.edge_level_db, self.cutoff]).all()
            or not 0 < self.fmin < self.fmax
        ):
            raise ValueError("Require finite 0 < fmin < fmax")
        if self.edge_level_db >= 0 or self.cutoff < 4:
            raise ValueError("Require negative edge level and at least four pulse widths")

    @property
    def frequency(self):
        return (self.fmin + self.fmax) / 2

    @property
    def tau(self):
        return np.sqrt(-self.edge_level_db * np.log(10) / 20) / (
            np.pi * (self.fmax - self.fmin) / 2
        )

    @property
    def delay(self):
        return self.cutoff * self.tau

    @property
    def duration(self):
        return 2 * self.delay

    @property
    def significant_frequency(self):
        """Upper frequency of the designed positive lobe at -40 dB amplitude."""
        return self.frequency + np.sqrt(2 * np.log(10)) / (np.pi * self.tau)

    def __call__(self, times):
        t = np.asarray(times)
        u = t - self.delay
        return np.where(
            (t >= 0) & (t <= self.duration),
            np.exp(-((u / self.tau) ** 2)) * np.cos(2 * np.pi * self.frequency * u),
            0.0,
        )

    def spectrum(self, frequencies):
        """Continuous untruncated real-pulse amplitude spectrum, normalized at fc."""
        f = np.asarray(frequencies)

        def a(z):
            return np.exp(-((np.pi * self.tau * (z - self.frequency)) ** 2)) + np.exp(
                -((np.pi * self.tau * (z + self.frequency)) ** 2)
            )

        return a(f) / a(self.frequency)
