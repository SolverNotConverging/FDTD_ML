"""Continuous single-object geometries for the second meshing curriculum.

The same shapes provide material membership and cut-edge intersections.  In
particular, polygon edges are never rasterized before they reach the FDTD solver.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .geometry import PEC, Material


def _finite_pair(value, name):
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    return pair


def _valid_material(material):
    if not isinstance(material, (Material, PEC)):
        raise ValueError("A dielectric Material or PEC is required")


def _merge_intervals(intervals, tolerance=1e-12):
    merged = []
    for left, right in sorted((float(a), float(b)) for a, b in intervals):
        if merged and left <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _cross2(left, right):
    return left[0] * right[1] - left[1] * right[0]


@dataclass(frozen=True)
class Ellipse:
    center: tuple[float, float]
    radii: tuple[float, float]
    angle: float
    material: Material | PEC

    def __post_init__(self):
        _finite_pair(self.center, "center")
        radii = _finite_pair(self.radii, "radii")
        if np.any(radii <= 0) or not np.isfinite(self.angle):
            raise ValueError("Ellipse requires positive radii and a finite angle")
        _valid_material(self.material)

    @property
    def bounds(self):
        cx, cy = self.center
        a, b = self.radii
        c, s = np.cos(self.angle), np.sin(self.angle)
        dx = np.hypot(a * c, b * s)
        dy = np.hypot(a * s, b * c)
        return cx - dx, cx + dx, cy - dy, cy + dy

    def contains(self, x, y):
        c, s = np.cos(self.angle), np.sin(self.angle)
        dx, dy = np.asarray(x) - self.center[0], np.asarray(y) - self.center[1]
        u, v = c * dx + s * dy, -s * dx + c * dy
        a, b = self.radii
        return (u / a) ** 2 + (v / b) ** 2 <= 1 + 1e-13

    def line_intervals(self, coordinate, axis, tolerance=0.0):
        """Analytic intersection of the filled conic with x or y constant."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        a, b = self.radii
        offset = float(coordinate) - self.center[1 - axis]
        if axis == 0:  # variable x, fixed y
            u0, u1, v0, v1 = c, s * offset, -s, c * offset
        elif axis == 1:  # variable y, fixed x
            u0, u1, v0, v1 = s, c * offset, c, -s * offset
        else:
            raise ValueError("axis must be 0 or 1")
        qa = (u0 / a) ** 2 + (v0 / b) ** 2
        qb = 2 * (u0 * u1 / a**2 + v0 * v1 / b**2)
        qc = (u1 / a) ** 2 + (v1 / b) ** 2 - 1
        discriminant = qb * qb - 4 * qa * qc
        if discriminant < -max(tolerance, 1e-14) * qa:
            return []
        half = np.sqrt(max(0.0, discriminant)) / (2 * qa)
        middle = self.center[axis] - qb / (2 * qa)
        return [(middle - half, middle + half)]


@dataclass(frozen=True)
class Polygon:
    vertices: tuple[tuple[float, float], ...]
    material: Material | PEC

    def __post_init__(self):
        points = np.asarray(self.vertices, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
            raise ValueError("Polygon requires at least three vertices")
        if not np.isfinite(points).all() or np.any(
            np.linalg.norm(np.roll(points, -1, 0) - points, axis=1) < 1e-12
        ):
            raise ValueError("Polygon contains invalid or repeated adjacent vertices")
        area = (
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)
            )
            / 2
        )
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if j in {i + 1, (i - 1) % len(points)} or (i == 0 and j == len(points) - 1):
                    continue
                a, b = points[i], points[(i + 1) % len(points)]
                c, d = points[j], points[(j + 1) % len(points)]
                ab, cd = b - a, d - c
                den = _cross2(ab, cd)
                if abs(den) < 1e-14:
                    continue
                t = _cross2(c - a, cd) / den
                u = _cross2(c - a, ab) / den
                if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
                    raise ValueError("Polygon must not self-intersect")
        if abs(area) < 1e-12:
            raise ValueError("Polygon area must be nonzero")
        _valid_material(self.material)

    @property
    def bounds(self):
        points = np.asarray(self.vertices)
        return (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )

    def contains(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        inside = np.zeros(x.shape, dtype=bool)
        boundary = np.zeros(x.shape, dtype=bool)
        points = np.asarray(self.vertices, dtype=float)
        for first, second in zip(points, np.roll(points, -1, axis=0)):
            ax, ay = first
            bx, by = second
            cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
            boundary |= (
                (np.abs(cross) <= 1e-12)
                & (x >= min(ax, bx) - 1e-12)
                & (x <= max(ax, bx) + 1e-12)
                & (y >= min(ay, by) - 1e-12)
                & (y <= max(ay, by) + 1e-12)
            )
            crosses = (ay > y) != (by > y)
            location = ax + (y - ay) * (bx - ax) / (by - ay if by != ay else 1.0)
            inside ^= crosses & (x < location)
        return inside | boundary

    def line_intervals(self, coordinate, axis, tolerance=0.0):
        """Pair exact segment crossings with a half-open vertex convention."""
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        points = np.asarray(self.vertices, dtype=float)
        variable, fixed = axis, 1 - axis
        crossings, boundary_segments = [], []
        for first, second in zip(points, np.roll(points, -1, axis=0)):
            a, b = first[fixed], second[fixed]
            if abs(a - coordinate) <= tolerance and abs(b - coordinate) <= tolerance:
                boundary_segments.append(
                    (min(first[variable], second[variable]), max(first[variable], second[variable]))
                )
            elif (a <= coordinate < b) or (b <= coordinate < a):
                fraction = (coordinate - a) / (b - a)
                crossings.append(first[variable] + fraction * (second[variable] - first[variable]))
        crossings.sort()
        if len(crossings) % 2:
            raise ValueError("Polygon line intersection has odd crossing count")
        return _merge_intervals(list(zip(crossings[::2], crossings[1::2])) + boundary_segments)


@dataclass(frozen=True)
class PolygonWithHoles:
    """One material region bounded by an outer polygon and void rings."""

    outer: Polygon
    holes: tuple[Polygon, ...]
    material: Material | PEC

    def __post_init__(self):
        if not isinstance(self.outer, Polygon) or not self.holes:
            raise ValueError("PolygonWithHoles requires an outer ring and at least one hole")
        _valid_material(self.material)
        if self.outer.material != self.material:
            raise ValueError("All rings must carry the parent material definition")
        for index, hole in enumerate(self.holes):
            if not isinstance(hole, Polygon):
                raise ValueError("Every hole must be a valid polygon ring")
            if hole.material != self.material:
                raise ValueError("All rings must carry the parent material definition")
            if any(not self.outer.contains(x, y) for x, y in hole.vertices):
                raise ValueError("Every hole vertex must lie inside the outer ring")
            if _rings_intersect(self.outer.vertices, hole.vertices):
                raise ValueError("A hole ring must not touch or intersect the outer ring")
            for previous in self.holes[:index]:
                if _rings_intersect(previous.vertices, hole.vertices):
                    raise ValueError("Hole rings must not touch or intersect")
                if previous.contains(*hole.vertices[0]) or hole.contains(*previous.vertices[0]):
                    raise ValueError("Hole rings must not nest")

    @property
    def bounds(self):
        return self.outer.bounds

    @property
    def rings(self):
        return (self.outer, *self.holes)

    @property
    def minimum_feature_size(self):
        """Conservative minimum of hole span and boundary-to-boundary thickness."""
        hole_spans = [
            min(hole.bounds[1] - hole.bounds[0], hole.bounds[3] - hole.bounds[2])
            for hole in self.holes
        ]
        ring_pairs = [(self.outer, hole) for hole in self.holes]
        ring_pairs.extend(
            (self.holes[i], self.holes[j])
            for i in range(len(self.holes))
            for j in range(i + 1, len(self.holes))
        )
        gaps = []
        for left, right in ring_pairs:
            a, b = np.asarray(left.vertices), np.asarray(right.vertices)
            for points, edges in ((a, b), (b, a)):
                for point in points:
                    for start, end in zip(edges, np.roll(edges, -1, axis=0)):
                        direction = end - start
                        fraction = np.clip(
                            (point - start) @ direction / (direction @ direction), 0, 1
                        )
                        gaps.append(float(np.linalg.norm(point - (start + fraction * direction))))
        return min((*hole_spans, *gaps))

    def contains(self, x, y):
        inside = self.outer.contains(x, y)
        for hole in self.holes:
            inside &= ~hole.contains(x, y)
        return inside

    def line_intervals(self, coordinate, axis, tolerance=0.0):
        intervals = self.outer.line_intervals(coordinate, axis, tolerance)
        for hole in self.holes:
            cuts = hole.line_intervals(coordinate, axis, tolerance)
            remaining = []
            for left, right in intervals:
                fragments = [(left, right)]
                for cut_left, cut_right in cuts:
                    next_fragments = []
                    for low, high in fragments:
                        if cut_right <= low or cut_left >= high:
                            next_fragments.append((low, high))
                        else:
                            if low < cut_left:
                                next_fragments.append((low, min(high, cut_left)))
                            if cut_right < high:
                                next_fragments.append((max(low, cut_right), high))
                    fragments = next_fragments
                remaining.extend(fragments)
            intervals = remaining
        return _merge_intervals(intervals, tolerance=max(1e-12, tolerance))


def _rings_intersect(left, right):
    """Closed-ring intersection check including endpoint and collinear contact."""
    first_ring, second_ring = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    for a, b in zip(first_ring, np.roll(first_ring, -1, axis=0)):
        ab = b - a
        for c, d in zip(second_ring, np.roll(second_ring, -1, axis=0)):
            cd = d - c
            denominator = _cross2(ab, cd)
            offset = c - a
            if abs(denominator) <= 1e-14:
                if abs(_cross2(offset, ab)) <= 1e-14:
                    projection = (c - a) @ ab / (ab @ ab)
                    other = (d - a) @ ab / (ab @ ab)
                    if max(min(projection, other), 0.0) <= min(max(projection, other), 1.0):
                        return True
                continue
            t = _cross2(offset, cd) / denominator
            u = _cross2(offset, ab) / denominator
            if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
                return True
    return False


def oriented_rectangle(center, width, height, angle, material):
    """Construct a rotated rectangle as an exact four-edge polygon."""
    center = _finite_pair(center, "center")
    if not np.isfinite([width, height, angle]).all() or width <= 0 or height <= 0:
        raise ValueError("Rectangle dimensions must be positive")
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * [width / 2, height / 2]
    c, s = np.cos(angle), np.sin(angle)
    vertices = local @ np.array([[c, s], [-s, c]]) + center
    return Polygon(tuple(map(tuple, vertices)), material)


@dataclass(frozen=True)
class SmoothLobed:
    """Star-shaped closed curve defined by a periodic cubic radial spline."""

    center: tuple[float, float]
    radii: tuple[float, ...]
    angle: float
    material: Material | PEC

    def __post_init__(self):
        _finite_pair(self.center, "center")
        values = np.asarray(self.radii, dtype=float)
        if (
            len(values) < 8
            or not np.isfinite(values).all()
            or np.any(values <= 0)
            or not np.isfinite(self.angle)
        ):
            raise ValueError("Smooth curve requires at least eight positive radial knots")
        _valid_material(self.material)
        if np.min(self._radius(np.linspace(0, 2 * np.pi, 4096, endpoint=False))) <= 0:
            raise ValueError("Radial spline must remain positive")

    @cached_property
    def spline(self):
        n = len(self.radii)
        return CubicSpline(
            np.linspace(0, 2 * np.pi, n + 1), np.r_[self.radii, self.radii[0]], bc_type="periodic"
        )

    def _radius(self, theta):
        return self.spline(np.mod(theta, 2 * np.pi))

    @property
    def bounds(self):
        theta = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
        radius = self._radius(theta)
        x = self.center[0] + radius * np.cos(theta + self.angle)
        y = self.center[1] + radius * np.sin(theta + self.angle)
        pad = 1e-6 * max(radius)
        return x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad

    def contains(self, x, y):
        dx = np.asarray(x) - self.center[0]
        dy = np.asarray(y) - self.center[1]
        theta = np.mod(np.arctan2(dy, dx) - self.angle, 2 * np.pi)
        return np.hypot(dx, dy) <= self._radius(theta) + 1e-13

    def line_intervals(self, coordinate, axis, tolerance=0.0):
        """Find spline/line cuts from all angular stationary intervals."""
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        spline = self.spline
        fixed = 1 - axis

        def value(theta):
            phase = theta + self.angle
            trig = np.sin(phase) if axis == 0 else np.cos(phase)
            return float(spline(theta) * trig + self.center[fixed] - coordinate)

        stationary = self.stationary_angles[axis]
        roots = []
        for left, right in zip(stationary[:-1], stationary[1:]):
            a, b = value(left), value(right)
            if abs(a) <= max(tolerance, 1e-13):
                roots.append(left)
            if a * b < 0:
                roots.append(brentq(value, left, right, xtol=1e-14))
        if abs(value(2 * np.pi)) <= max(tolerance, 1e-13):
            roots.append(2 * np.pi)
        positions = sorted(
            self.center[axis]
            + self._radius(np.asarray(roots))
            * (
                np.cos(np.asarray(roots) + self.angle)
                if axis == 0
                else np.sin(np.asarray(roots) + self.angle)
            )
        )
        positions = list(dict.fromkeys(round(float(p), 14) for p in positions))
        intervals = []
        for left, right in zip(positions[:-1], positions[1:]):
            midpoint = (left + right) / 2
            x, y = (midpoint, coordinate) if axis == 0 else (coordinate, midpoint)
            if self.contains(x, y):
                intervals.append((left, right))
        if len(positions) == 1:
            intervals.append((positions[0], positions[0]))
        return _merge_intervals(intervals)

    @cached_property
    def stationary_angles(self):
        spline = self.spline
        derivative = spline.derivative()
        sample = np.linspace(0, 2 * np.pi, 2049)
        result = []
        for axis in (0, 1):

            def slope(theta):
                phase = theta + self.angle
                trig = np.sin(phase) if axis == 0 else np.cos(phase)
                dtrig = np.cos(phase) if axis == 0 else -np.sin(phase)
                return float(derivative(theta) * trig + spline(theta) * dtrig)

            stationary = [0.0, 2 * np.pi]
            for left, right in zip(sample[:-1], sample[1:]):
                a, b = slope(left), slope(right)
                if a == 0:
                    stationary.append(left)
                elif a * b < 0:
                    stationary.append(brentq(slope, left, right, xtol=1e-14))
            result.append(tuple(sorted(set(stationary))))
        return tuple(result)
