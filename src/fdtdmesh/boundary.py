"""Explicit PEC discretization policy, independent of axis generation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryPolicy:
    mode: str = "conformal"
    max_fallback_fraction: float = 1.0
    on_fallback: str = "warn"

    def __post_init__(self):
        if self.mode not in ("conformal", "hybrid"):
            raise ValueError("Boundary mode must be conformal or hybrid")
        if self.on_fallback not in ("warn", "error"):
            raise ValueError("on_fallback must be warn or error")
        if not np.isfinite(self.max_fallback_fraction) or not 0 <= self.max_fallback_fraction <= 1:
            raise ValueError("max_fallback_fraction must be between zero and one")


class StaircaseFallbackWarning(UserWarning):
    """The discrete material representation locally differs from exact cuts."""
