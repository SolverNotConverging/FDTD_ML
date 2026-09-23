"""Geometry metrics and candidate meshes for localized sparse clusters."""

from itertools import combinations

import numpy as np

from .geometry import PEC, Circle, Material, Rectangle
from .meshing import density_axis


def _material(definition):
    if definition.get("kind") == "pec":
        return PEC()
    return Material(
        epsilon_r=float(definition["epsilon_r"]),
        sigma_e=float(definition.get("sigma_e_s_per_m", 0.0)),
    )


def sparse_scene_objects(scene):
    """Construct separated continuous circle/rectangle objects from a scene."""
    definitions = tuple(scene.get("objects", ()))
    if not 2 <= len(definitions) <= 4:
        raise ValueError("Sparse cluster scenes require two to four objects")
    objects = []
    for definition in definitions:
        shape = definition.get("shape")
        if shape == "circle":
            obj = Circle(
                tuple(float(value) for value in definition["center_m"]),
                float(definition["radius_m"]),
                _material(definition["material"]),
            )
        elif shape == "rectangle":
            obj = Rectangle(
                tuple(float(value) for value in definition["bounds_m"]),
                _material(definition["material"]),
            )
        else:
            raise ValueError(f"Unsupported sparse shape: {shape}")
        objects.append(obj)
    for left, right in combinations(objects, 2):
        _, _, distance = _closest_surface_points(left, right)
        if distance <= 1e-12:
            raise ValueError("Sparse pilot objects must be separated by a positive gap")
    return tuple(objects)


def circle_cluster_objects(scene):
    """Construct a circle-only cluster, retained for the M4.3A contract."""
    objects = sparse_scene_objects(scene)
    if not all(isinstance(obj, Circle) for obj in objects):
        raise ValueError("Circle cluster scenes support circles only")
    return objects


def _center(obj):
    if isinstance(obj, Circle):
        return np.asarray(obj.center, dtype=float)
    a, b, c, d = obj.bounds
    return np.asarray(((a + b) / 2, (c + d) / 2), dtype=float)


def _closest_surface_points(left, right):
    """Return closest boundary points and positive separation for supported shapes."""
    if isinstance(left, Circle) and isinstance(right, Circle):
        delta = np.asarray(right.center) - np.asarray(left.center)
        distance = float(np.linalg.norm(delta))
        if distance == 0:
            return np.asarray(left.center), np.asarray(right.center), -left.radius - right.radius
        direction = delta / distance
        first = np.asarray(left.center) + left.radius * direction
        second = np.asarray(right.center) - right.radius * direction
        return first, second, distance - left.radius - right.radius
    if isinstance(left, Rectangle) and isinstance(right, Rectangle):
        points = [[], []]
        for a0, a1, b0, b1 in (
            (left.bounds[0], left.bounds[1], right.bounds[0], right.bounds[1]),
            (left.bounds[2], left.bounds[3], right.bounds[2], right.bounds[3]),
        ):
            if a1 < b0:
                values = (a1, b0)
            elif b1 < a0:
                values = (a0, b1)
            else:
                middle = (max(a0, b0) + min(a1, b1)) / 2
                values = (middle, middle)
            points[0].append(values[0])
            points[1].append(values[1])
        first, second = np.asarray(points[0]), np.asarray(points[1])
        return first, second, float(np.linalg.norm(second - first))
    circle, rectangle = (left, right) if isinstance(left, Circle) else (right, left)
    a, b, c, d = rectangle.bounds
    rectangle_point = np.clip(np.asarray(circle.center), (a, c), (b, d))
    delta = rectangle_point - np.asarray(circle.center)
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        circle_point = np.asarray(circle.center)
        gap = -circle.radius
    else:
        circle_point = np.asarray(circle.center) + circle.radius * delta / distance
        gap = distance - circle.radius
    if isinstance(left, Circle):
        return circle_point, rectangle_point, gap
    return rectangle_point, circle_point, gap


def _union_length(intervals):
    ordered = sorted((float(left), float(right)) for left, right in intervals)
    total = 0.0
    current_left, current_right = ordered[0]
    for left, right in ordered[1:]:
        if left <= current_right:
            current_right = max(current_right, right)
        else:
            total += current_right - current_left
            current_left, current_right = left, right
    return total + current_right - current_left


def sparse_cluster_metrics(scene, *, domain=1.2):
    """Measure area, axis support, envelope, gaps, count, and PEC fraction."""
    if not np.isfinite(domain) or domain <= 0:
        raise ValueError("A positive finite domain is required")
    objects = sparse_scene_objects(scene)
    bounds = np.asarray([obj.bounds for obj in objects], dtype=float)
    if bounds[:, (0, 2)].min() < 0 or bounds[:, (1, 3)].max() > domain:
        raise ValueError("Every sparse object must remain inside the domain")
    gaps = [_closest_surface_points(left, right)[2] for left, right in combinations(objects, 2)]
    occupied_area = sum(
        np.pi * obj.radius**2
        if isinstance(obj, Circle)
        else (obj.bounds[1] - obj.bounds[0]) * (obj.bounds[3] - obj.bounds[2])
        for obj in objects
    )
    envelope = (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
    )
    pec_count = sum(isinstance(obj.material, PEC) for obj in objects)
    return {
        "occupied_area_fraction": float(occupied_area / domain**2),
        "projected_x_support_fraction": float(
            _union_length((obj.bounds[0], obj.bounds[1]) for obj in objects) / domain
        ),
        "projected_y_support_fraction": float(
            _union_length((obj.bounds[2], obj.bounds[3]) for obj in objects) / domain
        ),
        "cluster_envelope_m": list(envelope),
        "cluster_envelope_x_fraction": float((envelope[1] - envelope[0]) / domain),
        "cluster_envelope_y_fraction": float((envelope[3] - envelope[2]) / domain),
        "minimum_gap_m": min(gaps),
        "object_count": len(objects),
        "pec_fraction": pec_count / len(objects),
        "shape_topology": "+".join(
            sorted("circle" if isinstance(obj, Circle) else "rectangle" for obj in objects)
        ),
        "material_topology": "+".join(
            sorted("pec" if isinstance(obj.material, PEC) else "dielectric" for obj in objects)
        ),
    }


def _closest_pair(objects):
    return min(
        combinations(objects, 2),
        key=lambda pair: _closest_surface_points(*pair)[2],
    )


def sparse_circle_candidate_axes(domain, cells, scene, policy):
    """Backward-compatible alias for :func:`sparse_candidate_axes`."""
    return sparse_candidate_axes(domain, cells, scene, policy)


def sparse_candidate_axes(domain, cells, scene, policy):
    """Generate an exact-budget tensor mesh for a declared sparse-scene policy."""
    objects = sparse_scene_objects(scene)
    kind = policy["kind"]
    if kind == "uniform":
        axis = np.linspace(0.0, domain, int(cells) + 1)
        return axis.copy(), axis.copy()

    x_foci, y_foci = [], []
    if kind in {"interface", "hybrid"}:
        for obj in objects:
            a, b, c, d = obj.bounds
            width = float(policy["interface_width_factor"]) * min(b - a, d - c) / 2
            x_foci.extend((edge, width, policy["interface_weight"]) for edge in (a, b))
            y_foci.extend((edge, width, policy["interface_weight"]) for edge in (c, d))
    if kind in {"cluster", "hybrid"}:
        bounds = np.asarray([obj.bounds for obj in objects])
        center = (
            (bounds[:, 0].min() + bounds[:, 1].max()) / 2,
            (bounds[:, 2].min() + bounds[:, 3].max()) / 2,
        )
        widths = (
            max(bounds[:, 1].max() - bounds[:, 0].min(), 0.02),
            max(bounds[:, 3].max() - bounds[:, 2].min(), 0.02),
        )
        x_foci.append(
            (center[0], policy["cluster_width_factor"] * widths[0], policy["cluster_weight"])
        )
        y_foci.append(
            (center[1], policy["cluster_width_factor"] * widths[1], policy["cluster_weight"])
        )
    if kind in {"gap", "hybrid"}:
        left, right = _closest_pair(objects)
        surface_left, surface_right, gap = _closest_surface_points(left, right)
        midpoint = (surface_left + surface_right) / 2
        width = max(float(policy["gap_width_factor"]) * gap, float(policy["gap_min_width_m"]))
        x_foci.append((midpoint[0], width, policy["gap_weight"]))
        y_foci.append((midpoint[1], width, policy["gap_weight"]))
    if kind not in {"interface", "cluster", "gap", "hybrid"}:
        raise ValueError(f"Unsupported sparse candidate policy: {kind}")
    return density_axis(domain, cells, x_foci), density_axis(domain, cells, y_foci)
