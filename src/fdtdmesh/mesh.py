"""Exact-budget tensor-product grids; densities are constant within raster bins."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


class MeshInfeasibleError(ValueError):
    """The requested budget, anchors, or spacing constraints cannot be satisfied."""


def cell_count(value):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError("Cell counts must be positive integers")
    return int(value)


@dataclass(frozen=True)
class Mesh:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        for name in ("x", "y"):
            a = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
                raise ValueError("Mesh axes must be finite, strictly increasing 1-D coordinates")
            if a[0] != 0:
                raise ValueError("Mesh axes must begin at zero")
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    @property
    def Nx(self):
        return len(self.x) - 1

    @property
    def Ny(self):
        return len(self.y) - 1

    def diagnostics(self):
        result = {"Nx": self.Nx, "Ny": self.Ny}
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
    max_ratio: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.min_spacing) or self.min_spacing < 0:
            raise ValueError("min_spacing must be finite and nonnegative")
        if self.max_spacing is not None and (
            not np.isfinite(self.max_spacing)
            or self.max_spacing <= 0
            or self.max_spacing < self.min_spacing
        ):
            raise ValueError("max_spacing must be positive and >= min_spacing")
        if self.max_ratio is not None and (not np.isfinite(self.max_ratio) or self.max_ratio < 1):
            raise ValueError("max_ratio must be finite and >= 1")


def _allocate(masses, total, lower, upper):
    """Capped largest-remainder allocation, stable ties in coordinate order."""
    counts = lower.copy()
    if counts.sum() > total or upper.sum() < total:
        raise MeshInfeasibleError("Budget conflicts with anchor intervals and spacing bounds")
    while counts.sum() < total:
        active = np.flatnonzero(counts < upper)
        left = total - counts.sum()
        quotas = left * masses[active] / masses[active].sum()
        whole = np.minimum(np.floor(quotas).astype(int), upper[active] - counts[active])
        counts[active] += whole
        left = total - counts.sum()
        if left:
            order = active[np.argsort(-(quotas - np.floor(quotas)), kind="stable")]
            for i in order:
                if counts[i] < upper[i] and left:
                    counts[i] += 1
                    left -= 1
    return counts


def _project(lines, anchor_indices, anchors, constraints, length):
    n = len(lines) - 1
    lo = max(constraints.min_spacing / length, 1e-14)
    hi = constraints.max_spacing / length if constraints.max_spacing else 1.0
    target = np.diff(lines) / length
    eq = np.zeros((len(anchors) - 1, n))
    for row, (i, j) in enumerate(zip(anchor_indices[:-1], anchor_indices[1:])):
        eq[row, i:j] = 1
    rhs = np.diff(anchors) / length
    grading = np.zeros((2 * (n - 1), n))
    if constraints.max_ratio is not None:
        for i in range(n - 1):
            grading[2 * i, i : i + 2] = [-constraints.max_ratio, 1]
            grading[2 * i + 1, i : i + 2] = [1, -constraints.max_ratio]
    feasible = linprog(
        np.zeros(n),
        A_ub=grading,
        b_ub=np.zeros(len(grading)),
        A_eq=eq,
        b_eq=rhs,
        bounds=[(lo, hi)] * n,
        method="highs",
    )
    if not feasible.success:
        raise MeshInfeasibleError(
            "Spacing/grading infeasible for the allocated anchor intervals; "
            "relax constraints or change the budget/density"
        )
    cumulative = np.tril(np.ones((n, n)))

    def objective(h):
        delta = cumulative @ (h - target)
        return 0.5 * delta @ delta, cumulative.T @ delta

    result = minimize(
        objective,
        feasible.x,
        jac=True,
        method="SLSQP",
        bounds=Bounds(lo, hi),
        constraints=[LinearConstraint(eq, rhs, rhs), LinearConstraint(grading, -np.inf, 0)],
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    if not result.success:
        raise MeshInfeasibleError(f"Mesh constraint projection failed: {result.message}")
    out = np.r_[0, np.cumsum(result.x)] * length
    out[anchor_indices] = anchors
    h = np.diff(out) / length
    tol = 1e-9
    if (
        np.any(h <= 0)
        or h.min() < lo - tol
        or h.max() > hi + tol
        or np.max(grading @ h, initial=0) > tol
        or (
            constraints.max_ratio is not None
            and n > 1
            and max(np.max(h[1:] / h[:-1]), np.max(h[:-1] / h[1:]))
            > constraints.max_ratio * (1 + 1e-8)
        )
    ):
        raise MeshInfeasibleError("Projected mesh failed final constraint verification")
    return out


def axis_mesh(length, count, density, anchors=(), constraints=None):
    count = cell_count(count)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Axis length must be finite and positive")
    rho = np.asarray(density, dtype=float)
    if rho.ndim != 1 or not rho.size or not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("Density must be a finite, positive 1-D array")
    rho = rho / rho.max()  # Avoid overflow; scale has no effect on quantiles.
    if np.any(rho == 0):
        raise ValueError("Density dynamic range exceeds floating-point precision")
    a = np.unique(np.r_[0.0, anchors, length])
    if not np.isfinite(a).all() or a[0] < 0 or a[-1] > length:
        raise ValueError("Anchors must be finite and inside the domain")
    if len(a) - 1 > count:
        raise MeshInfeasibleError("There must be at least one cell between each pair of anchors")
    c = constraints or AxisConstraints()
    edges = np.unique(np.r_[np.linspace(0, length, len(rho) + 1), a])
    mid = (edges[:-1] + edges[1:]) / 2
    values = rho[np.minimum((mid / length * len(rho)).astype(int), len(rho) - 1)]
    cdf = np.r_[0, np.cumsum(np.diff(edges) * values)]
    if np.any(np.diff(cdf) <= 0):
        raise MeshInfeasibleError("Density/anchor dynamic range cannot be resolved in float64")
    am = np.interp(a, edges, cdf)
    spans = np.diff(a)
    lower = np.ones(len(spans), dtype=int)
    upper = np.full(len(spans), count, dtype=int)
    if c.max_spacing:
        lower = np.maximum(lower, np.ceil(spans / c.max_spacing - 1e-12).astype(int))
    if c.min_spacing:
        upper = np.minimum(upper, np.floor(spans / c.min_spacing + 1e-12).astype(int))
    if np.any(upper < lower):
        raise MeshInfeasibleError("Anchor separation violates spacing bounds")
    counts = _allocate(np.diff(am), count, lower, upper)
    chunks = [
        np.interp(np.linspace(am[i], am[i + 1], k + 1)[:-1], cdf, edges)
        for i, k in enumerate(counts)
    ]
    lines = np.r_[np.concatenate(chunks), length]
    indices = np.r_[0, np.cumsum(counts)]
    lines[indices] = a
    if c.min_spacing or c.max_spacing or c.max_ratio:
        lines = _project(lines, indices, a, c, length)
    if len(lines) != count + 1 or np.any(np.diff(lines) <= 0):
        raise MeshInfeasibleError("Mesh has unresolvable or duplicate lines")
    return lines


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
):
    return Mesh(
        axis_mesh(Lx, Nx, rho_x, x_anchors, x_constraints),
        axis_mesh(Ly, Ny, rho_y, y_anchors, y_constraints),
    )
