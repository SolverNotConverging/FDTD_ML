"""Exact-budget tensor-product grids; densities are constant within raster bins."""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_RATIO = 1.4
MESH_POLICY = {
    "version": 2,
    "max_ratio": DEFAULT_MAX_RATIO,
    "objective": "normalized_line_L1",
    "allocation": "joint_anchor_MILP",
}


class MeshInfeasibleError(ValueError):
    """The requested budget, anchors, or spacing constraints cannot be satisfied."""


class MeshOptimizationError(RuntimeError):
    """Optimization did not finish; distinct from proven mesh infeasibility."""


def cell_count(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or int(value) != value
        or value < 1
    ):
        raise ValueError("Cell counts must be positive integers")
    return int(value)


@dataclass(frozen=True)
class Mesh:
    x: np.ndarray
    y: np.ndarray
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        for name in ("x", "y"):
            a = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
                raise ValueError("Mesh axes must be finite, strictly increasing 1-D coordinates")
            if a[0] != 0:
                raise ValueError("Mesh axes must begin at zero")
            validate_spacing(a, AxisConstraints())
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    @property
    def Nx(self):
        return len(self.x) - 1

    @property
    def Ny(self):
        return len(self.y) - 1

    def diagnostics(self):
        result = {"Nx": self.Nx, "Ny": self.Ny, **self.metadata}
        for name in ("x", "y"):
            h = np.diff(getattr(self, name))
            ratios = h[1:] / h[:-1]
            result[f"d{name}_min"] = float(h.min())
            result[f"d{name}_max"] = float(h.max())
            result[f"{name}_grading"] = float(
                max(1, ratios.max(initial=1), (1 / ratios).max(initial=1))
            )
        return result


@dataclass(frozen=True)
class AxisConstraints:
    min_spacing: float = 0.0
    max_spacing: float | None = None
    max_ratio: float = DEFAULT_MAX_RATIO

    def __post_init__(self):
        if not np.isfinite(self.min_spacing) or self.min_spacing < 0:
            raise ValueError("min_spacing must be finite and nonnegative")
        if self.max_spacing is not None and (
            not np.isfinite(self.max_spacing)
            or self.max_spacing <= 0
            or self.max_spacing < self.min_spacing
        ):
            raise ValueError("max_spacing must be positive and >= min_spacing")
        if (
            self.max_ratio is None
            or not np.isfinite(self.max_ratio)
            or not 1 <= self.max_ratio <= DEFAULT_MAX_RATIO
        ):
            raise ValueError("Grading is mandatory: max_ratio must lie in [1, 1.4]")


@dataclass(frozen=True)
class AxisCollar:
    """Symmetric fixed PML collars counted inside the total cell budget."""

    cells: int = 0
    thickness: float = 0.0

    def __post_init__(self):
        if self.cells == 0 and self.thickness == 0:
            return
        object.__setattr__(self, "cells", cell_count(self.cells))
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("PML collar thickness must be finite and positive")

    def fixed_lines(self, length, count):
        if 2 * self.cells >= count or 2 * self.thickness >= length:
            raise MeshInfeasibleError("PML collars leave no interior cells or physical width")
        if not self.cells:
            return {0: 0.0, count: length}
        left = np.linspace(0, self.thickness, self.cells + 1)
        right = np.linspace(length - self.thickness, length, self.cells + 1)
        return {**dict(enumerate(left)), **{count - self.cells + i: x for i, x in enumerate(right)}}


def adjacent_ratio(h):
    return max(1.0, np.max(h[1:] / h[:-1], initial=1), np.max(h[:-1] / h[1:], initial=1))


def validate_spacing(lines, constraints):
    h = np.diff(lines)
    tol = 1e-8
    if np.any(h <= 0) or adjacent_ratio(h) > constraints.max_ratio * (1 + tol):
        raise MeshInfeasibleError(
            f"Mesh violates maximum adjacent spacing ratio {constraints.max_ratio}"
        )
    if h.min() < constraints.min_spacing * (1 - tol):
        raise MeshInfeasibleError("Mesh violates minimum spacing")
    if constraints.max_spacing is not None and h.max() > constraints.max_spacing * (1 + tol):
        raise MeshInfeasibleError("Mesh violates maximum spacing")


def axis_mesh(
    length,
    count,
    density,
    anchors=(),
    constraints=None,
    *,
    collar=None,
    return_diagnostics=False,
    time_limit=30.0,
):
    from .mesh_projection import project_axis

    return project_axis(
        length,
        count,
        density,
        anchors,
        constraints,
        collar=collar,
        return_diagnostics=return_diagnostics,
        time_limit=time_limit,
    )


def density_mesh(
    Lx,
    Ly,
    Nx,
    Ny,
    rho_x,
    rho_y,
    *,
    x_anchors=(),
    y_anchors=(),
    x_constraints=None,
    y_constraints=None,
    x_collar=None,
    y_collar=None,
    time_limit=30.0,
):
    x, xd = axis_mesh(
        Lx,
        Nx,
        rho_x,
        x_anchors,
        x_constraints,
        collar=x_collar,
        return_diagnostics=True,
        time_limit=time_limit,
    )
    y, yd = axis_mesh(
        Ly,
        Ny,
        rho_y,
        y_anchors,
        y_constraints,
        collar=y_collar,
        return_diagnostics=True,
        time_limit=time_limit,
    )
    return Mesh(
        x, y, {**{f"x_{k}": v for k, v in xd.items()}, **{f"y_{k}": v for k, v in yd.items()}}
    )


def projected_density(lines, bins, *, collar=None):
    """Equal mass per learned cell integrated into CNN bins; detached teacher target."""
    bins = cell_count(bins)
    collar = collar or AxisCollar()
    length = lines[-1]
    interior = np.asarray(lines)[collar.cells : len(lines) - collar.cells]
    if len(interior) < 2:
        raise ValueError("No learned cells")
    edges = np.linspace(0, length, bins + 1)
    cdf = np.interp(edges, interior, np.linspace(0, 1, len(interior)), left=0, right=1)
    return np.diff(cdf)
