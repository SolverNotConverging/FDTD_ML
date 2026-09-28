"""Exact 2D PEC silhouettes for engineering-scale meshing studies.

Aircraft are top-view planforms; the telescope is a side-view dish and support.
The TMz solver extrudes every shape along z, so these are not 3D CAD models.
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import C0
from ..geometry import Geometry

ENGINEERED_SHAPES = ("swept_aircraft", "propeller_aeroplane", "radio_telescope")


def make_engineered_geometry(
    name, *, frequency=1e9, scale=1.0, incidence_deg=0.0, pec_struts=False
):
    """Return an unbounded exact CSG recipe, measured in free-space wavelengths.

    The telescope feed is held by electrically transparent support unless
    pec_struts=True explicitly adds two thin PEC struts.
    """
    if name not in ENGINEERED_SHAPES:
        raise ValueError(f"Unknown engineered shape {name!r}")
    if not np.isfinite([frequency, scale, incidence_deg]).all() or frequency <= 0 or scale <= 0:
        raise ValueError("Frequency/scale must be positive and incidence finite")
    unit = C0 / frequency * scale
    geometry = Geometry()

    def polygon(points):
        nonlocal geometry
        geometry, _ = geometry.added(
            "polygon", tuple((unit * x, unit * y) for x, y in points), "PEC"
        )

    def rectangle(x0, x1, y0, y1):
        nonlocal geometry
        geometry, _ = geometry.added("rectangle", tuple(unit * v for v in (x0, x1, y0, y1)), "PEC")

    def mirrored_outline(right):
        return right + [(-x, y) for x, y in right[-2:0:-1]]

    def thin_support(start, end, width):
        dx, dy = end[0] - start[0], end[1] - start[1]
        half = width / (2 * math.hypot(dx, dy))
        normal = (-dy * half, dx * half)
        polygon(
            [
                (start[0] + normal[0], start[1] + normal[1]),
                (end[0] + normal[0], end[1] + normal[1]),
                (end[0] - normal[0], end[1] - normal[1]),
                (start[0] - normal[0], start[1] - normal[1]),
            ]
        )

    if name == "swept_aircraft":
        right = [
            (0, 1.25),
            (0.10, 1.05),
            (0.13, 0.45),
            (0.22, 0.36),
            (1.35, -0.14),
            (1.45, -0.25),
            (1.32, -0.36),
            (0.22, -0.05),
            (0.17, -0.82),
            (0.65, -0.98),
            (0.65, -1.08),
            (0.16, -1.02),
            (0.13, -1.17),
            (0, -1.20),
        ]
        polygon(mirrored_outline(right))
        # Two nacelles overlap the wing and remain part of the exact CSG union.
        for side in (-1, 1):
            polygon(
                [
                    (side * 0.48, 0.19),
                    (side * 0.57, 0.14),
                    (side * 0.60, -0.18),
                    (side * 0.47, -0.16),
                ]
            )
    elif name == "propeller_aeroplane":
        right = [
            (0, 1.15),
            (0.13, 1.07),
            (0.15, 0.45),
            (0.65, 0.35),
            (1.15, 0.30),
            (1.20, 0.20),
            (1.15, 0.08),
            (0.15, 0.07),
            (0.13, -0.60),
            (0.60, -0.78),
            (0.60, -0.88),
            (0.12, -0.86),
            (0.10, -1.03),
            (0, -1.08),
        ]
        polygon(mirrored_outline(right))
        rectangle(-0.48, 0.48, 1.02, 1.12)  # broad propeller blade
        rectangle(-0.045, 0.045, 1.01, 1.23)  # hub joins the nose
    else:
        # Finite-thickness parabolic reflector, y = 0.43 x^2, with open mouth.
        x = np.linspace(-1.05, 1.05, 33)
        upper = [(float(v), float(0.43 * v * v)) for v in x]
        lower = [(float(v), float(0.43 * v * v - 0.065)) for v in x[::-1]]
        polygon(upper + lower)
        rectangle(-0.065, 0.065, -0.93, -0.04)  # azimuth mount
        rectangle(-0.28, 0.28, -1.03, -0.88)  # foundation
        if pec_struts:
            thin_support((-0.95, 0.385), (-0.025, 0.625), 0.035)
            thin_support((0.95, 0.385), (0.025, 0.625), 0.035)
        geometry, _ = geometry.added("circle", (0, 0.63 * unit, 0.075 * unit), "PEC")

    if incidence_deg:
        geometry = geometry.rotated(-math.radians(incidence_deg), origin=(0, 0))
    return geometry
