"""Canonical analytic benchmark geometries.

The recipes in this module are expressed as offsets from the domain centre.
They remain continuous geometry until the simulation creates its mesh.
"""

from __future__ import annotations

import math

from ..geometry import Geometry

SHAPES = (
    "circle",
    "ellipse",
    "rectangle",
    "triangle",
    "pentagon",
    "convex",
    "l_shape",
    "u_shape",
    "star",
    "wifi",
    "moon",
    "sun",
)


def _add(g: Geometry, kind: str, values, material: str = "PEC") -> Geometry:
    return g.added(kind, values, material)[0]


def _point(center, scale: float, offset):
    return (center[0] + scale * offset[0], center[1] + scale * offset[1])


def _polygon(g, center, scale, offsets, material="PEC"):
    return _add(g, "polygon", tuple(_point(center, scale, p) for p in offsets), material)


def make_geometry(name, *, size=None, scale, incidence_deg=0.0):
    """Build one of the canonical benchmark shapes.

    ``size`` is the domain size in metres and ``scale`` converts canonical
    offsets to metres.  Incidence rotates the completed shape about the domain
    centre, in degrees.
    """
    if name not in SHAPES:
        raise ValueError(f"Unknown benchmark shape {name!r}; expected one of {SHAPES}")
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise ValueError("scale must be a positive finite number") from None
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite number")
    try:
        incidence_deg = float(incidence_deg)
    except (TypeError, ValueError):
        raise ValueError("incidence_deg must be finite") from None
    if not math.isfinite(incidence_deg):
        raise ValueError("incidence_deg must be finite")

    g = Geometry(None if size is None else tuple(size))
    center = (0.0, 0.0) if size is None else (g.size[0] / 2.0, g.size[1] / 2.0)

    if name == "circle":
        g = _add(g, "circle", (*center, 0.65 * scale))
    elif name == "ellipse":
        g = _add(g, "ellipse", (*center, 0.85 * scale, 0.5 * scale, 0.0))
    elif name == "rectangle":
        g = _add(
            g,
            "rectangle",
            (
                center[0] - 0.75 * scale,
                center[0] + 0.75 * scale,
                center[1] - 0.5 * scale,
                center[1] + 0.5 * scale,
            ),
        )
    elif name == "triangle":
        g = _polygon(g, center, scale, ((-0.8, -0.55), (0.8, -0.55), (0, 0.8)))
    elif name == "pentagon":
        vertices = tuple(
            (
                0.85 * math.cos(math.pi / 2 + i * 2 * math.pi / 5),
                0.85 * math.sin(math.pi / 2 + i * 2 * math.pi / 5),
            )
            for i in range(5)
        )
        g = _polygon(g, center, scale, vertices)
    elif name == "convex":
        g = _polygon(
            g,
            center,
            scale,
            ((-0.8, -0.35), (-0.35, -0.75), (0.55, -0.6), (0.85, 0.2), (0.1, 0.8), (-0.65, 0.5)),
        )
    elif name == "l_shape":
        g = _polygon(
            g,
            center,
            scale,
            ((-0.7, -0.7), (0.7, -0.7), (0.7, -0.25), (-0.25, -0.25), (-0.25, 0.7), (-0.7, 0.7)),
        )
    elif name == "u_shape":
        g = _polygon(
            g,
            center,
            scale,
            (
                (-0.75, -0.7),
                (0.75, -0.7),
                (0.75, 0.7),
                (0.3, 0.7),
                (0.3, -0.2),
                (-0.3, -0.2),
                (-0.3, 0.7),
                (-0.75, 0.7),
            ),
        )
    elif name == "star":
        vertices = tuple(
            (
                (0.9 if i % 2 == 0 else 0.48) * math.cos(math.pi / 2 + i * math.pi / 5),
                (0.9 if i % 2 == 0 else 0.48) * math.sin(math.pi / 2 + i * math.pi / 5),
            )
            for i in range(10)
        )
        g = _polygon(g, center, scale, vertices)
    elif name == "moon":
        g = _add(g, "circle", (*center, 0.85 * scale))
        g = _add(g, "circle", (*_point(center, scale, (0.35, 0.15)), 0.75 * scale), "air")
    elif name == "sun":
        g = _add(g, "circle", (*center, 0.48 * scale))
        for i in range(8):
            angle = i * math.pi / 4
            g = _polygon(
                g,
                center,
                scale,
                (
                    (0.4 * math.cos(angle - 0.22), 0.4 * math.sin(angle - 0.22)),
                    (0.9 * math.cos(angle), 0.9 * math.sin(angle)),
                    (0.4 * math.cos(angle + 0.22), 0.4 * math.sin(angle + 0.22)),
                ),
            )
    elif name == "wifi":
        wifi_center = _point(center, scale, (0.0, -0.3))
        for radius, air_radius in ((0.95, 0.76), (0.59, 0.40)):
            g = _add(g, "circle", (*wifi_center, radius * scale))
            g = _add(g, "circle", (*wifi_center, air_radius * scale), "air")
        g = _add(
            g,
            "rectangle",
            (
                center[0] - 1.1 * scale,
                center[0] + 1.1 * scale,
                center[1] - 1.4 * scale,
                center[1] - 0.1 * scale,
            ),
            "air",
        )
        g = _add(g, "circle", (*_point(center, scale, (0.0, -0.45)), 0.12 * scale))

    if incidence_deg:
        g = g.rotated(-math.radians(incidence_deg), origin=center)
    return g
