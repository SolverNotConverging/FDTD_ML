"""Exact unbounded PEC shape recipes; lengths in metres, orientation in degrees."""

import numpy as np

from ..constants import C0
from .engineered_shapes import ENGINEERED_SHAPES, make_engineered_geometry
from .shapes import SHAPES
from .shapes import make_geometry as _simple


def make_geometry(name, *, scale, orientation_deg=0.0):
    """Return exact geometry centred near the origin; orientation rotates the object."""
    if not np.isfinite([scale, orientation_deg]).all() or scale <= 0:
        raise ValueError("scale must be positive and orientation finite")
    if name in ENGINEERED_SHAPES:
        return make_engineered_geometry(
            name, frequency=C0, scale=scale, incidence_deg=-orientation_deg
        )
    return _simple(name, scale=scale, incidence_deg=-orientation_deg)


__all__ = ["SHAPES", "ENGINEERED_SHAPES", "make_geometry"]
