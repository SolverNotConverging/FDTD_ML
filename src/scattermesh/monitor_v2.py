"""Numerical contour placement shared by v2 FDTD qualification."""

import numpy as np


def widest_non_pml_monitor_bounds(grid, object_bounds, pml_thickness):
    """Choose the widest legal NF2FF contour with one vacuum cell around objects."""
    thickness = np.broadcast_to(np.asarray(pml_thickness, dtype=float), (2,))
    if not np.isfinite(thickness).all() or np.any(thickness <= 0):
        raise ValueError("Positive finite PML thickness is required")
    bounds = np.asarray(tuple(object_bounds), dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 4 or not np.isfinite(bounds).all():
        raise ValueError("Finite object bounds are required")

    indices = []
    for axis, pml, lower, upper in (
        (grid.x, thickness[0], bounds[:, 0].min(), bounds[:, 1].max()),
        (grid.y, thickness[1], bounds[:, 2].min(), bounds[:, 3].max()),
    ):
        length = axis[-1]
        left = [index for index in range(1, len(axis) - 1) if axis[index - 1] > pml]
        right = [index for index in range(1, len(axis) - 1) if axis[index + 1] < length - pml]
        if not left or not right:
            raise ValueError("PML leaves no legal monitor interpolation stencil")
        i0, i1 = min(left), max(right)
        if not (i0 + 1 < i1 - 1 and axis[i0 + 1] < lower < upper < axis[i1 - 1]):
            raise ValueError("Grid and PML cannot enclose every scatterer with a full vacuum cell")
        indices.append((i0, i1))
    (i0, i1), (j0, j1) = indices
    return float(grid.x[i0]), float(grid.x[i1]), float(grid.y[j0]), float(grid.y[j1])
