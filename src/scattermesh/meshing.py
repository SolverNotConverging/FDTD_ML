"""Deterministic tensor-grid candidates from separable refinement densities."""

import numpy as np


def density_axis(length, cells, foci=(), *, samples=16385):
    """Return exact density quantiles for Gaussian ``(center, width, weight)`` foci.

    This is a candidate generator, not a learned mesher. A weight adds local line
    density without anchoring a line to the focus. The caller still chooses and
    validates a grading cap through :class:`scattermesh.Grid`.
    """
    if (
        not np.isfinite(length)
        or length <= 0
        or isinstance(cells, bool)
        or int(cells) != cells
        or cells < 4
        or isinstance(samples, bool)
        or int(samples) != samples
        or samples < 257
    ):
        raise ValueError("Positive length, cells >=4, and samples >=257 required")
    foci = tuple(tuple(focus) for focus in foci)
    for focus in foci:
        if len(focus) != 3 or not np.isfinite(focus).all():
            raise ValueError("Each focus needs finite center, width, and weight")
        center, width, weight = focus
        if not 0 <= center <= length or width <= 0 or weight < 0:
            raise ValueError("Focus centers must be in-domain with positive width and weight")
    coordinate = np.linspace(0, length, int(samples))
    density = np.ones_like(coordinate)
    for center, width, weight in foci:
        density += weight * np.exp(-(((coordinate - center) / width) ** 2))
    cumulative = np.r_[
        0.0,
        np.cumsum((density[1:] + density[:-1]) * np.diff(coordinate) / 2),
    ]
    axis = np.interp(np.linspace(0, cumulative[-1], int(cells) + 1), cumulative, coordinate)
    axis[0], axis[-1] = 0.0, float(length)
    return axis


def circular_interface_axes(length, cells, center, radius, *, width, weight):
    """Project circular-interface importance onto legal x/y tensor-grid densities."""
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or radius <= 0:
        raise ValueError("Finite 2D center and positive radius required")
    x_foci = [(center[0] - radius, width, weight), (center[0] + radius, width, weight)]
    y_foci = [(center[1] - radius, width, weight), (center[1] + radius, width, weight)]
    return density_axis(length, cells, x_foci), density_axis(length, cells, y_foci)
