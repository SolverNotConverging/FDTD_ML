"""Grid-independent corner anchors for the procedural catalog's CSG boundaries."""

import numpy as np


def geometry_anchors(geometry):
    scene = geometry.to_scene()
    points, segments, circles = [], [], []
    for kind, _, data in scene.primitives:
        if kind == "polygon":
            points.extend(data)
            segments.extend(zip(data, np.roll(data, -1, axis=0)))
        elif kind == "circle":
            circles.append(data)

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for i, (a, b) in enumerate(segments):
        v = b - a
        for c, d in segments[i + 1 :]:
            w = d - c
            den = cross(v, w)
            if abs(den) > 1e-15 * np.linalg.norm(v) * np.linalg.norm(w):
                t, u = cross(c - a, w) / den, cross(c - a, v) / den
                if 0 <= t <= 1 and 0 <= u <= 1:
                    points.append(a + t * v)
        for cx, cy, r in circles:
            u = a - np.array([cx, cy])
            aa, bb, cc = v @ v, 2 * (u @ v), u @ u - r * r
            disc = bb * bb - 4 * aa * cc
            if disc >= 0:
                for t in ((-bb - np.sqrt(disc)) / (2 * aa), (-bb + np.sqrt(disc)) / (2 * aa)):
                    if 0 <= t <= 1:
                        points.append(a + t * v)
    for i, (x, y, r) in enumerate(circles):
        c = np.array([x, y])
        for xx, yy, rr in circles[i + 1 :]:
            v = np.array([xx, yy]) - c
            d = np.linalg.norm(v)
            if d > 0 and abs(r - rr) <= d <= r + rr:
                along = (r * r - rr * rr + d * d) / (2 * d)
                height = np.sqrt(max(0, r * r - along * along))
                mid = c + along * v / d
                perpendicular = np.array([-v[1], v[0]]) / d
                points.extend([mid + height * perpendicular, mid - height * perpendicular])
    if not points:
        return (np.array([]), np.array([]))
    points = np.asarray(points)
    # Filter hidden CSG vertices using continuous occupancy, never a Yee raster.
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False) + 0.017
    eps = min(geometry.size) * 1e-8
    material = scene.contains(
        points[:, 0, None] + eps * np.cos(phi), points[:, 1, None] + eps * np.sin(phi)
    )
    points = points[np.any(material, axis=1) & ~np.all(material, axis=1)]
    axes = []
    for axis in (0, 1):
        values = np.sort(points[:, axis])
        # Collapse only roundoff duplicate coordinates, not physical feature gaps.
        keep = (
            np.r_[True, np.diff(values) > 64 * np.finfo(float).eps * geometry.size[axis]]
            if len(values)
            else []
        )
        axes.append(values[keep])
    return tuple(axes)
