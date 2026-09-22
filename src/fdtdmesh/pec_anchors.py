"""Optional tensor-grid anchors for PEC faces and geometric features."""

import numpy as np


def _bounds(kind, data):
    if kind == "polygon":
        return np.min(data, axis=0), np.max(data, axis=0)
    if kind == "circle":
        x, y, r = data
        return np.array([x - r, y - r]), np.array([x + r, y + r])
    return np.array([np.min(data.x), np.min(data.y)]), np.array([np.max(data.x), np.max(data.y)])


def add_pec_anchors(scene, *, mode="axis_aligned", include_overrides=True):
    """Add exact face anchors, optionally polygon vertices and circle extrema.

    Cartesian mesh lines do not conform to slanted/curved boundaries. Feature
    anchors are local-resolution constraints, not a conformal PEC discretization.
    Later ordinary objects intersecting an earlier PEC body's bounding box are
    included conservatively because they can expose new PEC cutout faces.
    Close mandatory coordinates are never snapped, merged, or silently removed.
    """
    if mode not in ("axis_aligned", "features"):
        raise ValueError("PEC anchor mode must be axis_aligned or features")
    before = {axis: set(getattr(scene, f"{axis}_anchors")) for axis in ("x", "y")}
    earlier_pec = []
    selected = []
    for index, (kind, material, data) in enumerate(scene.primitives):
        lo, hi = _bounds(kind, data)
        override = (
            include_overrides
            and material.kind != "PEC"
            and any(np.all(np.maximum(lo, a) < np.minimum(hi, b)) for a, b in earlier_pec)
        )
        if material.kind == "PEC" or override:
            added = False
            if kind == "line":
                for axis, values in (("x", data.x), ("y", data.y)):
                    for value in np.atleast_1d(values):
                        scene.add_anchor(axis, float(value))
                added = True
            elif kind == "polygon":
                if mode == "features":
                    for axis, values in zip(("x", "y"), np.asarray(data).T):
                        for value in values:
                            scene.add_anchor(axis, float(value))
                    added = True
                else:
                    for a, b in zip(data, np.roll(data, -1, axis=0)):
                        # Require exactly axis-aligned geometry; never rotate/snap it.
                        if a[0] == b[0] or a[1] == b[1]:
                            for axis, values in zip(("x", "y"), np.array([a, b]).T):
                                for value in values:
                                    scene.add_anchor(axis, float(value))
                            added = True
            elif mode == "features":
                x, y, r = data
                for axis, center in (("x", x), ("y", y)):
                    for value in (center - r, center, center + r):
                        scene.add_anchor(axis, float(value))
                added = True
            if added:
                selected.append(dict(primitive=index, kind=kind, ordinary_override=bool(override)))
        if material.kind == "PEC":
            earlier_pec.append((lo, hi))
    return dict(
        mode=mode,
        include_overrides=include_overrides,
        primitives=selected,
        added={
            axis: sorted(getattr(scene, f"{axis}_anchors") - before[axis]) for axis in ("x", "y")
        },
    )
