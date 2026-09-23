"""Tensor-product Yee grids without mandatory geometry/source/monitor anchors."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    max_ratio: float | None = 3.0

    def __post_init__(self):
        if self.max_ratio is not None and (not np.isfinite(self.max_ratio) or self.max_ratio < 1):
            raise ValueError("max_ratio must be >=1 or None")
        for axis in "xy":
            a = np.array(getattr(self, axis), dtype=np.float64, copy=True)
            if a.ndim != 1 or len(a) < 5 or not np.isfinite(a).all() or a[0] != 0:
                raise ValueError("Grid axes need >=5 finite nodes beginning at zero")
            h = np.diff(a)
            if np.any(h <= 0):
                raise ValueError("Grid spacing must be positive")
            ratio = max(float(np.max(h[1:] / h[:-1])), float(np.max(h[:-1] / h[1:])))
            if self.max_ratio is not None and ratio > self.max_ratio * (1 + 1e-12):
                raise ValueError(f"Grid grading {ratio:g} exceeds selected limit {self.max_ratio}")
            a.flags.writeable = False
            object.__setattr__(self, axis, a)

    @property
    def shape(self):
        return len(self.x), len(self.y)

    @property
    def centers(self):
        return (self.x[:-1] + self.x[1:]) / 2, (self.y[:-1] + self.y[1:]) / 2

    def diagnostics(self):
        result = {"Nx": len(self.x) - 1, "Ny": len(self.y) - 1, "max_ratio_limit": self.max_ratio}
        for axis in "xy":
            h = np.diff(getattr(self, axis))
            result.update(
                {
                    f"d{axis}_min": float(h.min()),
                    f"d{axis}_max": float(h.max()),
                    f"{axis}_grading": float(max(np.max(h[1:] / h[:-1]), np.max(h[:-1] / h[1:]))),
                }
            )
        return result


def focused_axis(length, cells, *, center=None, width=None, strength=0):
    """Smooth, anchor-free density quantiles for numerical experiments, not a teacher."""
    if (
        not np.isfinite(length)
        or length <= 0
        or isinstance(cells, bool)
        or int(cells) != cells
        or cells < 4
    ):
        raise ValueError("Positive length and integer cell count >=4 required")
    center = length / 2 if center is None else center
    width = length / 8 if width is None else width
    if not np.isfinite([center, width, strength]).all() or width <= 0 or strength < 0:
        raise ValueError("Invalid focus parameters")
    if strength == 0:
        return np.linspace(0, length, cells + 1)
    x = np.linspace(0, length, 16385)
    rho = 1 + strength * np.exp(-(((x - center) / width) ** 2))
    cumulative = np.r_[0, np.cumsum((rho[1:] + rho[:-1]) * np.diff(x) / 2)]
    return np.interp(np.linspace(0, cumulative[-1], cells + 1), cumulative, x)
