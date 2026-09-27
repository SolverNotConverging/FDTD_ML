"""Immutable continuous geometry. No grid-dependent state lives in this module."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene import Scene2D


@dataclass(frozen=True)
class Shape:
    id: int
    kind: str
    material: str
    parameters: tuple
    name: str | None = None

    def __post_init__(self):
        if self.kind not in ("circle", "ellipse", "rectangle", "polygon") or self.material not in (
            "PEC",
            "air",
        ):
            raise ValueError("Unsupported geometry kind or material")
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Geometry handle must be a nonnegative integer")
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("Geometry name must be a string or None")
        values = (
            tuple(tuple(map(float, p)) for p in self.parameters)
            if self.kind == "polygon"
            else tuple(map(float, self.parameters))
        )
        if (
            self.kind != "polygon"
            and len(values) != {"circle": 3, "ellipse": 5, "rectangle": 4}[self.kind]
        ):
            raise ValueError("Invalid primitive parameters")
        object.__setattr__(self, "parameters", values)


@dataclass(frozen=True)
class Geometry:
    size: tuple | None = None
    shapes: tuple = ()
    next_id: int = 0

    def __post_init__(self):
        if self.size is not None:
            object.__setattr__(self, "size", tuple(float(v) for v in self.size))
            if len(self.size) != 2:
                raise ValueError("size must contain two lengths in metres")
            Scene2D(*self.size)
        object.__setattr__(self, "shapes", tuple(self.shapes))
        if any(not isinstance(s, Shape) for s in self.shapes):
            raise ValueError("Geometry requires Shape records")
        ids = [s.id for s in self.shapes]
        if (
            isinstance(self.next_id, bool)
            or not isinstance(self.next_id, int)
            or self.next_id < 0
            or len(set(ids)) != len(ids)
            or any(i >= self.next_id for i in ids)
        ):
            raise ValueError("Invalid geometry handles")
        self.to_scene()

    def to_scene(self):
        # Unbounded design geometry uses a working scale only for roundoff tests.
        # contains/material_bounds do not clip to the Scene2D rectangle.
        scale = 1.0
        if self.size is None and self.shapes:
            spans = []
            for shape in self.shapes:
                p = np.asarray(shape.parameters)
                spans.append(
                    float(np.ptp(p, axis=0).max())
                    if shape.kind == "polygon"
                    else max(p[2:4]) * 2
                    if shape.kind == "ellipse"
                    else p[2] * 2
                    if shape.kind == "circle"
                    else max(p[1] - p[0], p[3] - p[2])
                )
            scale = max(spans)
            # Subtracting a small radius from a far-away centre loses precision
            # on the absolute-coordinate scale, even for a tiny shape.
            scale = max(
                scale,
                *(
                    float(
                        np.max(
                            np.abs(
                                s.parameters[:2]
                                if s.kind in ("circle", "ellipse")
                                else s.parameters
                            )
                        )
                    )
                    for s in self.shapes
                ),
            )
        scene = Scene2D(*(self.size or (scale, scale)), validate_primitives=False)
        for s in self.shapes:
            p = s.parameters
            if s.kind == "circle":
                scene.add_circle(p[:2], p[2], pec=s.material == "PEC")
            elif s.kind == "ellipse":
                scene.add_ellipse(p[:2], p[2:4], p[4], pec=s.material == "PEC")
            elif s.kind == "rectangle":
                scene.add_rectangle(p[:2], p[2:], pec=s.material == "PEC")
            elif s.kind == "polygon":
                scene.add_polygon(p, pec=s.material == "PEC")
            else:
                raise ValueError(f"Unknown geometry type {s.kind}")
        bounds = scene.material_bounds()
        if bounds is not None and self.size is not None:
            x0, x1, y0, y1 = bounds
            if not (0 < x0 <= x1 < self.size[0] and 0 < y0 <= y1 < self.size[1]):
                raise ValueError(
                    f"Final PEC geometry bounds {bounds} m must lie strictly inside domain {self.size} m"
                )
        return scene

    @property
    def bounds(self):
        return self.to_scene().material_bounds()

    def translated(self, offset, *, size=None):
        """Translate the complete CSG recipe and optionally change domain size."""
        dx, dy = offset
        shapes = []
        for shape in self.shapes:
            p = shape.parameters
            if shape.kind in ("circle", "ellipse"):
                values = (p[0] + dx, p[1] + dy, *p[2:])
            elif shape.kind == "rectangle":
                values = (p[0] + dx, p[1] + dx, p[2] + dy, p[3] + dy)
            else:
                values = tuple((x + dx, y + dy) for x, y in p)
            shapes.append(Shape(shape.id, shape.kind, shape.material, values, shape.name))
        return Geometry(self.size if size is None else size, tuple(shapes), self.next_id)

    def added(self, kind, parameters, material="PEC", name=None):
        if material not in ("PEC", "air"):
            raise ValueError("material must be 'PEC' or 'air'")
        p = (
            tuple(tuple(map(float, v)) for v in parameters)
            if kind == "polygon"
            else tuple(map(float, parameters))
        )
        shape = Shape(self.next_id, kind, material, p, name)
        result = Geometry(self.size, (*self.shapes, shape), self.next_id + 1)
        return result, shape.id

    def removed(self, handle):
        if not any(s.id == handle for s in self.shapes):
            raise KeyError(f"No geometry handle {handle}")
        return Geometry(self.size, tuple(s for s in self.shapes if s.id != handle), self.next_id)

    def contains(self, x, y):
        return self.to_scene().contains(x, y)

    def rotated(self, angle, *, origin=None):
        """Return the same ordered continuous shapes rigidly rotated in radians."""
        if not np.isfinite(angle):
            raise ValueError("Rotation must be finite")
        if origin is None:
            bounds = self.bounds
            origin = (
                np.array(self.size) / 2
                if self.size
                else (
                    np.array([bounds[0] + bounds[1], bounds[2] + bounds[3]]) / 2
                    if bounds
                    else np.zeros(2)
                )
            )
        else:
            origin = np.asarray(origin, float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Invalid rotation origin")
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])

        def transform(p):
            return (np.asarray(p) - origin) @ rotation.T + origin

        shapes = []
        for shape in self.shapes:
            p = shape.parameters
            kind = shape.kind
            if kind in ("circle", "ellipse"):
                values = (*transform(p[:2]), *p[2:])
                if kind == "ellipse":
                    values = (*values[:4], p[4] + angle)
            else:
                if kind == "rectangle":
                    p = ((p[0], p[2]), (p[1], p[2]), (p[1], p[3]), (p[0], p[3]))
                kind, values = "polygon", transform(p)
            shapes.append(Shape(shape.id, kind, shape.material, values, shape.name))
        return Geometry(self.size, tuple(shapes), self.next_id)

    def as_dict(self):
        return dict(
            schema=1,
            units="m",
            size=self.size,
            next_id=self.next_id,
            shapes=[
                dict(
                    id=s.id, kind=s.kind, material=s.material, parameters=s.parameters, name=s.name
                )
                for s in self.shapes
            ],
        )

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != 1 or data.get("units") != "m":
            raise ValueError("Unsupported geometry schema or units")
        return cls(
            None if data["size"] is None else tuple(data["size"]),
            tuple(Shape(**s) for s in data["shapes"]),
            data["next_id"],
        )

    @property
    def geometry_id(self):
        return hashlib.sha256(
            json.dumps(
                self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def rasterize(self, shape=(256, 256), *, channels=("occupancy",), bounds=None):
        """Pixel-centre samples, array order (channels, y, x), independent of any mesh."""
        if len(shape) != 2 or any(isinstance(n, bool) or int(n) != n or n < 1 for n in shape):
            raise ValueError("shape must contain positive integer (height, width)")
        h, w = map(int, shape)
        if bounds is None:
            bounds = (0.0, self.size[0], 0.0, self.size[1]) if self.size else self.bounds
        if bounds is None:
            raise ValueError("Empty unbounded geometry needs explicit raster bounds")
        x0, x1, y0, y1 = bounds
        if (
            not np.isfinite([x0, x1, y0, y1]).all()
            or not x0 < x1
            or not y0 < y1
            or (
                self.size is not None
                and (not 0 <= x0 < x1 <= self.size[0] or not 0 <= y0 < y1 <= self.size[1])
            )
        ):
            raise ValueError("Raster bounds must lie in the domain")
        x, y = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w, y0 + (np.arange(h) + 0.5) * (y1 - y0) / h
        xx, yy = np.meshgrid(x, y)
        available = {
            "occupancy": self.contains(xx, yy).astype(np.float32),
            "x": (xx / self.size[0] if self.size else (xx - x0) / (x1 - x0)).astype(np.float32),
            "y": (yy / self.size[1] if self.size else (yy - y0) / (y1 - y0)).astype(np.float32),
        }
        if not channels or any(c not in available for c in channels):
            raise ValueError("Supported channels: occupancy, x, y")
        return np.stack([available[c] for c in channels])
