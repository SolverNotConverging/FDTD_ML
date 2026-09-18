"""One continuous, SI-coordinate scene shared by the CNN and every Yee grid."""

from dataclasses import dataclass

import numpy as np


def raster_index(position, length, count):
    """Choose the containing pixel; round normalized boundary ties consistently."""
    return min(count - 1, max(0, int(round(position / length * count, 12))))


@dataclass(frozen=True)
class Material:
    name: str
    epsilon_r: float = 1.0
    mu_r: float = 1.0
    sigma_e: float = 0.0
    kind: str = "ordinary"

    def __post_init__(self):
        if self.kind not in ("ordinary", "PEC"):
            raise ValueError("Only ordinary isotropic materials and PEC are supported")
        if (
            not np.isfinite([self.epsilon_r, self.mu_r, self.sigma_e]).all()
            or self.epsilon_r <= 0
            or self.mu_r <= 0
            or self.sigma_e < 0
        ):
            raise ValueError("Materials require epsilon_r, mu_r > 0 and sigma_e >= 0")


@dataclass(frozen=True)
class Probe:
    kind: str
    x: float | tuple
    y: float | tuple
    samples: int | None = None


class Scene2D:
    def __init__(self, Lx, Ly):
        if not np.isfinite([Lx, Ly]).all() or min(Lx, Ly) <= 0:
            raise ValueError("Domain lengths must be finite and positive")
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.materials = {"vacuum": Material("vacuum"), "PEC": Material("PEC", kind="PEC")}
        self.primitives = []
        self.x_anchors, self.y_anchors = {0.0, self.Lx}, {0.0, self.Ly}
        self.sources, self.receivers = [], []
        self.pml = None

    def add_material(self, name, **kwargs):
        if name in self.materials:
            raise ValueError(f"Material {name!r} already exists")
        material = Material(name, **kwargs)
        self.materials[name] = material
        return material

    def _material(self, material):
        if isinstance(material, Material):
            return material
        return self.materials[material]

    def _points(self, points):
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points > [self.Lx, self.Ly])
        ):
            raise ValueError("Coordinates must be finite physical metres inside the domain")
        return points

    def add_anchor(self, axis, position):
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        limit = self.Lx if axis == "x" else self.Ly
        if not np.isfinite(position) or not 0 <= position <= limit:
            raise ValueError("Anchor outside domain")
        getattr(self, f"{axis}_anchors").add(float(position))

    def add_rectangle(self, material, x_position, y_position):
        x0, x1 = x_position
        y0, y1 = y_position
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Rectangle ranges must be increasing")
        self.add_polygon(material, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def add_circle(self, material, center, radius):
        x, y = center
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Circle radius must be positive")
        self._points([(x - radius, y - radius), (x + radius, y + radius)])
        self.primitives.append(("circle", self._material(material), (x, y, radius)))

    def add_triangle(self, material, vertices):
        if len(vertices) != 3:
            raise ValueError("Triangle requires three vertices")
        self.add_polygon(material, vertices)

    def add_polygon(self, material, vertices):
        v = self._points(vertices).copy()
        if (
            len(v) < 3
            or abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)))
            < 1e-15 * self.Lx * self.Ly
        ):
            raise ValueError("Polygon must have at least three vertices and nonzero area")
        self.primitives.append(("polygon", self._material(material), v))

    def add_pec_line(self, *, x, y):
        probe = self.make_probe("line", x, y)
        self.primitives.append(("line", self.materials["PEC"], probe))

    def make_probe(self, kind, x, y):
        kind = "line" if kind == "line-soft" else kind
        xs, ys = np.ndim(x) == 0, np.ndim(y) == 0
        if kind == "point" and xs and ys:
            self._points([(x, y)])
            return Probe(kind, float(x), float(y))
        if kind != "line" or xs == ys:
            raise ValueError(
                "Use point x/y scalars or an axis-aligned line with one coordinate pair"
            )
        start, stop = y if xs else x
        if start >= stop:
            raise ValueError("Line endpoints must be increasing")
        self._points([(x, start), (x, stop)] if xs else [(start, y), (stop, y)])
        # Endpoints also anchor the finite line extent, so it cannot disappear.
        for axis, values in (("x", [x] if xs else x), ("y", y if xs else [y])):
            for value in values:
                self.add_anchor(axis, value)
        return Probe(kind, float(x) if xs else tuple(x), tuple(y) if xs else float(y))

    def sample(self, x, y, *, raster=False):
        """Point sample materials. Last inserted primitive wins, including PEC."""
        X, Y = np.meshgrid(x, y, indexing="ij")
        eps, mu, sigma = np.ones_like(X), np.ones_like(X), np.zeros_like(X)
        pec = np.zeros(X.shape, dtype=bool)
        tol = 1e-12 * max(self.Lx, self.Ly)
        for kind, mat, data in self.primitives:
            if kind == "circle":
                cx, cy, r = data
                mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
            elif kind == "polygon":
                mask = np.zeros(X.shape, dtype=bool)
                boundary = mask.copy()
                for (xa, ya), (xb, yb) in zip(data, np.roll(data, -1, axis=0)):
                    cross = (X - xa) * (yb - ya) - (Y - ya) * (xb - xa)
                    boundary |= (
                        (abs(cross) <= tol * np.hypot(xb - xa, yb - ya))
                        & (X >= min(xa, xb) - tol)
                        & (X <= max(xa, xb) + tol)
                        & (Y >= min(ya, yb) - tol)
                        & (Y <= max(ya, yb) + tol)
                    )
                    if ya != yb:
                        mask ^= ((ya > Y) != (yb > Y)) & (X < (xb - xa) * (Y - ya) / (yb - ya) + xa)
                mask |= boundary
            else:
                mask = np.zeros(X.shape, dtype=bool)
                if np.ndim(data.x) == 0:
                    fixed = (
                        raster_index(data.x, self.Lx, len(x))
                        if raster
                        else np.argmin(abs(np.asarray(x) - data.x))
                    )
                    if raster or abs(x[fixed] - data.x) <= tol:
                        mask[fixed, :] = (np.asarray(y) >= data.y[0] - tol) & (
                            np.asarray(y) <= data.y[1] + tol
                        )
                else:
                    fixed = (
                        raster_index(data.y, self.Ly, len(y))
                        if raster
                        else np.argmin(abs(np.asarray(y) - data.y))
                    )
                    if raster or abs(y[fixed] - data.y) <= tol:
                        mask[:, fixed] = (np.asarray(x) >= data.x[0] - tol) & (
                            np.asarray(x) <= data.x[1] + tol
                        )
            eps[mask], mu[mask], sigma[mask] = mat.epsilon_r, mat.mu_r, mat.sigma_e
            pec[mask] = mat.kind == "PEC"
        return eps, mu, sigma, pec


def probe_indices(probe, mesh):
    if probe.kind == "point":
        return np.array([[np.argmin(abs(mesh.x - probe.x)), np.argmin(abs(mesh.y - probe.y))]])
    if np.ndim(probe.x) == 0:
        i = np.argmin(abs(mesh.x - probe.x))
        js = np.flatnonzero((mesh.y >= probe.y[0]) & (mesh.y <= probe.y[1]))
        return np.column_stack((np.full(len(js), i), js))
    j = np.argmin(abs(mesh.y - probe.y))
    ii = np.flatnonzero((mesh.x >= probe.x[0]) & (mesh.x <= probe.x[1]))
    return np.column_stack((ii, np.full(len(ii), j)))
