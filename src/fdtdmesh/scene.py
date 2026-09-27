"""Continuous vacuum/PEC geometry, including analytic circles and vacuum holes."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene2D:
    Lx: float
    Ly: float
    primitives: list = field(default_factory=list)

    def __post_init__(self):
        if not np.isfinite([self.Lx, self.Ly]).all() or min(self.Lx, self.Ly) <= 0:
            raise ValueError("Domain lengths must be finite and positive")

    def _points(self, points):
        p = np.asarray(points, dtype=float)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or not np.isfinite(p).all()
            or np.any(p <= 0)
            or np.any(p >= [self.Lx, self.Ly])
        ):
            raise ValueError("Geometry must lie strictly inside the domain")
        return p

    def add_circle(self, center, radius, *, pec=True):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Radius must be positive")
        x, y = center
        self._points([(x - radius, y - radius), (x + radius, y + radius)])
        self.primitives.append(("circle", bool(pec), (float(x), float(y), float(radius))))
        return self

    def add_polygon(self, vertices, *, pec=True):
        v = self._points(vertices).copy()
        if (
            len(v) < 3
            or abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)))
            < 1e-14 * self.Lx * self.Ly
        ):
            raise ValueError("Polygon must have nonzero area")

        def cross(u, w):
            return u[0] * w[1] - u[1] * w[0]

        for i in range(len(v)):
            a, b = v[i], v[(i + 1) % len(v)]
            if np.linalg.norm(b - a) == 0:
                raise ValueError("Repeated polygon vertex")
            for j in range(i + 2, len(v)):
                if (j + 1) % len(v) == i:
                    continue
                c, d = v[j], v[(j + 1) % len(v)]
                if (
                    cross(b - a, c - a) * cross(b - a, d - a) <= 0
                    and cross(d - c, a - c) * cross(d - c, b - c) <= 0
                ):
                    if np.all(
                        np.maximum(np.minimum(a, b), np.minimum(c, d))
                        <= np.minimum(np.maximum(a, b), np.maximum(c, d))
                    ):
                        raise ValueError("Polygon must be simple")
        self.primitives.append(("polygon", bool(pec), v))
        return self

    def add_rectangle(self, x, y, *, pec=True):
        x0, x1 = x
        y0, y1 = y
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Rectangle ranges must increase")
        return self.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], pec=pec)

    def contains(self, x, y):
        """PEC closure of the ordered, regularized material regions.

        Boundary points are classified from analytic directional limits, so two
        touching air cuts do not leave a fictitious zero-thickness PEC sheet.
        No mesh spacing or finite-distance dilation enters the construction.
        """
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        result = np.zeros(x.shape, bool)
        tol = 8 * np.finfo(float).eps * max(self.Lx, self.Ly)
        on_boundary = np.zeros(x.shape, bool)
        for kind, metal, data in self.primitives:
            if kind == "circle":
                cx, cy, r = data
                distance = (x - cx) ** 2 + (y - cy) ** 2
                on_boundary |= abs(distance - r * r) <= tol * r
                inside = distance <= r * r + tol * r if metal else distance < r * r - tol * r
            else:
                inside, boundary = np.zeros(x.shape, bool), np.zeros(x.shape, bool)
                for (xa, ya), (xb, yb) in zip(data, np.roll(data, -1, axis=0)):
                    cross = (x - xa) * (yb - ya) - (y - ya) * (xb - xa)
                    boundary |= (
                        (abs(cross) <= tol * np.hypot(xb - xa, yb - ya))
                        & (x >= min(xa, xb) - tol)
                        & (x <= max(xa, xb) + tol)
                        & (y >= min(ya, yb) - tol)
                        & (y <= max(ya, yb) + tol)
                    )
                    if ya != yb:
                        inside ^= ((ya > y) != (yb > y)) & (
                            x < (xb - xa) * (y - ya) / (yb - ya) + xa
                        )
                inside = inside | boundary if metal else inside & ~boundary
                on_boundary |= boundary
            result[inside] = metal
        for index in np.flatnonzero(on_boundary):
            result.flat[index] = self._boundary_material(
                float(x.flat[index]), float(y.flat[index]), tol
            )
        return result

    def _boundary_material(self, x, y, tol):
        angles, curved_tangents = [], []
        for kind, _, data in self.primitives:
            if kind == "circle":
                cx, cy, r = data
                if abs((x - cx) ** 2 + (y - cy) ** 2 - r * r) <= tol * r:
                    a = np.arctan2(y - cy, x - cx) + np.pi / 2
                    angles.extend([a, a + np.pi])
                    curved_tangents.extend([a, a + np.pi])
            else:
                for (xa, ya), (xb, yb) in zip(data, np.roll(data, -1, axis=0)):
                    if (
                        abs((x - xa) * (yb - ya) - (y - ya) * (xb - xa))
                        <= tol * np.hypot(xb - xa, yb - ya)
                        and min(xa, xb) - tol <= x <= max(xa, xb) + tol
                        and min(ya, yb) - tol <= y <= max(ya, yb) + tol
                    ):
                        a = np.arctan2(yb - ya, xb - xa)
                        angles.extend([a, a + np.pi])
        angles = np.unique(np.mod(angles, 2 * np.pi))
        if not len(angles):
            return False
        # Open sectors cover polygon corners, including arbitrarily narrow wedges.
        # Circle tangents additionally detect cusps between touching circular cuts.
        probes = (angles + np.r_[angles[1:], angles[0] + 2 * np.pi]) / 2
        probes = np.r_[probes, curved_tangents]
        dx, dy = np.cos(probes), np.sin(probes)
        material = np.zeros(len(probes), bool)
        angular_tol = 32 * np.finfo(float).eps
        for kind, metal, data in self.primitives:
            if kind == "circle":
                cx, cy, r = data
                delta = r * r - (x - cx) ** 2 - (y - cy) ** 2
                inside = np.full(len(probes), delta > 0)
                if abs(delta) <= tol * r:
                    inside = (x - cx) * dx + (y - cy) * dy < -angular_tol * r
            else:
                inside = np.zeros(len(probes), bool)
                along = np.zeros(len(probes), bool)
                for (xa, ya), (xb, yb) in zip(data, np.roll(data, -1, axis=0)):
                    if ya != yb:
                        above_a = np.full(len(probes), ya > y) if abs(ya - y) > tol else dy < 0
                        above_b = np.full(len(probes), yb > y) if abs(yb - y) > tol else dy < 0
                        v = (xb - xa) * (y - ya) / (yb - ya) + xa - x
                        right = (
                            np.full(len(probes), v > 0)
                            if abs(v) > tol
                            else (xb - xa) * dy / (yb - ya) - dx > angular_tol
                        )
                        inside ^= (above_a != above_b) & right
                    ex, ey = xb - xa, yb - ya
                    length = np.hypot(ex, ey)
                    if (
                        abs((x - xa) * ey - (y - ya) * ex) <= tol * length
                        and min(xa, xb) - tol <= x <= max(xa, xb) + tol
                        and min(ya, yb) - tol <= y <= max(ya, yb) + tol
                    ):
                        tangent = abs(dx * ey - dy * ex) <= angular_tol * length
                        if np.hypot(x - xa, y - ya) <= tol:
                            tangent &= dx * ex + dy * ey > 0
                        if np.hypot(x - xb, y - yb) <= tol:
                            tangent &= dx * ex + dy * ey < 0
                        along |= tangent
                inside[along] = not metal
            material[inside] = metal
        return bool(np.any(material))

    def bounds(self):
        bounds = []
        for kind, _, data in self.primitives:
            if kind == "circle":
                x, y, r = data
                bounds.append((x - r, x + r, y - r, y + r))
            else:
                bounds.append(
                    (data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max())
                )
        return bounds

    def vacuum_intervals(self, axis, fixed):
        """Exact intersections and ordered Boolean overlays along an axis line."""
        length = self.Lx if axis == 0 else self.Ly
        breaks = [0.0, length]
        for kind, _, data in self.primitives:
            if kind == "circle":
                delta = data[2] ** 2 - (fixed - data[1 - axis]) ** 2
                if delta > 0:
                    root = np.sqrt(delta)
                    breaks.extend([data[axis] - root, data[axis] + root])
            else:
                for a, b in zip(data, np.roll(data, -1, axis=0)):
                    da = b[1 - axis] - a[1 - axis]
                    if da != 0:
                        t = (fixed - a[1 - axis]) / da
                        if 0 <= t <= 1:
                            breaks.append(a[axis] + t * (b[axis] - a[axis]))
                    elif fixed == a[1 - axis]:
                        breaks.extend([a[axis], b[axis]])
        breaks = np.unique(np.clip(breaks, 0, length))
        mid = (breaks[:-1] + breaks[1:]) / 2
        metal = self.contains(mid, fixed) if axis == 0 else self.contains(fixed, mid)
        return [(a, b) for a, b, m in zip(breaks[:-1], breaks[1:], metal) if not m]

    def as_dict(self):
        return {
            "Lx": self.Lx,
            "Ly": self.Ly,
            "primitives": [(k, p, np.asarray(d).tolist()) for k, p, d in self.primitives],
        }
