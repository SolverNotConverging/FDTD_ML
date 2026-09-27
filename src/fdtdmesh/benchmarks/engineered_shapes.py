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
        geometry, _ = geometry.added(
            "rectangle", tuple(unit * v for v in (x0, x1, y0, y1)), "PEC"
        )

    def mirrored_outline(right):
        return right + [(-x, y) for x, y in right[-2:0:-1]]

    def thin_support(start, end, width):
        dx, dy = end[0] - start[0], end[1] - start[1]
        half = width / (2 * math.hypot(dx, dy))
        normal = (-dy * half, dx * half)
        polygon([
            (start[0] + normal[0], start[1] + normal[1]),
            (end[0] + normal[0], end[1] + normal[1]),
            (end[0] - normal[0], end[1] - normal[1]),
            (start[0] - normal[0], start[1] - normal[1]),
        ])

    if name == "swept_aircraft":
        right = [
            (0, 1.25), (.10, 1.05), (.13, .45), (.22, .36),
            (1.35, -.14), (1.45, -.25), (1.32, -.36), (.22, -.05),
            (.17, -.82), (.65, -.98), (.65, -1.08), (.16, -1.02),
            (.13, -1.17), (0, -1.20),
        ]
        polygon(mirrored_outline(right))
        # Two nacelles overlap the wing and remain part of the exact CSG union.
        for side in (-1, 1):
            polygon([(side * .48, .19), (side * .57, .14),
                     (side * .60, -.18), (side * .47, -.16)])
    elif name == "propeller_aeroplane":
        right = [
            (0, 1.15), (.13, 1.07), (.15, .45), (.65, .35),
            (1.15, .30), (1.20, .20), (1.15, .08), (.15, .07),
            (.13, -.60), (.60, -.78), (.60, -.88), (.12, -.86),
            (.10, -1.03), (0, -1.08),
        ]
        polygon(mirrored_outline(right))
        rectangle(-.48, .48, 1.02, 1.12)  # broad propeller blade
        rectangle(-.045, .045, 1.01, 1.23)  # hub joins the nose
    else:
        # Finite-thickness parabolic reflector, y = 0.43 x^2, with open mouth.
        x = np.linspace(-1.05, 1.05, 33)
        upper = [(float(v), float(.43 * v * v)) for v in x]
        lower = [(float(v), float(.43 * v * v - .065)) for v in x[::-1]]
        polygon(upper + lower)
        rectangle(-.065, .065, -.93, -.04)  # azimuth mount
        rectangle(-.28, .28, -1.03, -.88)  # foundation
        if pec_struts:
            thin_support((-.95, .385), (-.025, .625), .035)
            thin_support((.95, .385), (.025, .625), .035)
        geometry, _ = geometry.added("circle", (0, .63 * unit, .075 * unit), "PEC")

    if incidence_deg:
        geometry = geometry.rotated(-math.radians(incidence_deg), origin=(0, 0))
    return geometry
