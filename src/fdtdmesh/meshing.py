"""Advanced axis preparation options for the public mesh API."""

from dataclasses import dataclass

import numpy as np

from .mesh import AxisConstraints, cell_count


@dataclass(frozen=True)
class MeshOptions:
    constraints: AxisConstraints | None = None
    anchors: tuple | None = None
    anchor_assignment: str = "local"
    time_limit: float = 30.0
    max_passes: int = 20
    hybrid_repair_passes: int = 3

    def __post_init__(self):
        if self.constraints is not None and not isinstance(self.constraints, AxisConstraints):
            raise TypeError("constraints must be AxisConstraints")
        if self.anchor_assignment not in ("local", "joint", "fixed"):
            raise ValueError("Invalid anchor_assignment")
        if not np.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        object.__setattr__(self, "max_passes", cell_count(self.max_passes))
        n = self.hybrid_repair_passes
        if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n < 0:
            raise ValueError("hybrid_repair_passes must be a nonnegative integer")
