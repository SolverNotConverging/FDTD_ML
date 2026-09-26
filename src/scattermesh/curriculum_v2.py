"""Versioned continuous-geometry scenes, including legacy singles and collections."""

import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_edt

from .constants import C0
from .geometry import PEC, Circle, Material, Rectangle
from .geometry_v2 import Ellipse, Polygon, PolygonWithHoles, SmoothLobed, oriented_rectangle

DOMAIN = 1.2
FAMILIES = (
    "circle",
    "ellipse",
    "rectangle",
    "triangle",
    "convex_polygon",
    "concave_polygon",
    "star",
    "smooth_lobed",
)

ENGINEERING_FAMILIES = ("aircraft", "ship", "vehicle")


def _material(definition):
    if definition["kind"] == "pec":
        return PEC()
    if definition["kind"] != "dielectric":
        raise ValueError("Material kind must be dielectric or pec")
    return Material(float(definition["epsilon_r"]), float(definition.get("sigma_e_s_per_m", 0)))


def object_from_scene(scene):
    """Construct the continuous object used by both solver and rasterizer."""
    if "shape" not in scene:
        raise ValueError(
            "object_from_scene accepts one geometry definition; use objects_from_scene"
        )
    if scene["shape"] == "polygon_with_holes" and scene.get("schema_version") != 4:
        raise ValueError("polygon_with_holes requires scene schema version 4")
    return object_from_definition(scene)


def object_from_definition(definition):
    """Construct one continuous object from a legacy scene or collection member."""
    material = _material(definition["material"])
    shape = definition["shape"]
    if shape == "circle":
        return Circle(tuple(definition["center_m"]), float(definition["radius_m"]), material)
    if shape == "ellipse":
        return Ellipse(
            tuple(definition["center_m"]),
            tuple(definition["radii_m"]),
            float(definition["angle_rad"]),
            material,
        )
    if shape == "rectangle":
        return Rectangle(tuple(definition["bounds_m"]), material)
    if shape == "rotated_rectangle":
        return oriented_rectangle(
            definition["center_m"],
            definition["width_m"],
            definition["height_m"],
            definition["angle_rad"],
            material,
        )
    if shape in {"triangle", "convex_polygon", "concave_polygon", "star"}:
        return Polygon(tuple(tuple(point) for point in definition["vertices_m"]), material)
    if shape == "polygon_with_holes":
        outer = Polygon(tuple(tuple(point) for point in definition["outer_ring_m"]), material)
        holes = tuple(
            Polygon(tuple(tuple(point) for point in ring), material)
            for ring in definition["holes_m"]
        )
        return PolygonWithHoles(outer, holes, material)
    if shape == "smooth_lobed":
        return SmoothLobed(
            tuple(definition["center_m"]),
            tuple(definition["radii_knots_m"]),
            float(definition["angle_rad"]),
            material,
        )
    raise ValueError(f"Unsupported v2 shape: {shape}")


def objects_from_scene(scene):
    """Return a legacy single object or a versioned scene object collection."""
    definitions = scene.get("objects")
    if definitions is None:
        return (object_from_scene(scene),)
    if not isinstance(definitions, list) or not 1 <= len(definitions) <= 10:
        raise ValueError("A collection scene requires between one and ten object definitions")
    identifiers = [item.get("object_id") for item in definitions]
    if any(identifier is None for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Every collection object requires a unique object_id")
    if (
        any(item.get("shape") == "polygon_with_holes" for item in definitions)
        and scene.get("schema_version") != 4
    ):
        raise ValueError("polygon_with_holes requires scene schema version 4")
    return tuple(object_from_definition(item) for item in definitions)


def scene_material_kind(scene):
    """Summarize a homogeneous legacy scene or a mixed material collection."""
    kinds = {
        "pec" if isinstance(obj.material, PEC) else "dielectric"
        for obj in objects_from_scene(scene)
    }
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _object_area(obj):
    if isinstance(obj, Circle):
        return np.pi * obj.radius**2
    if isinstance(obj, Ellipse):
        return np.pi * obj.radii[0] * obj.radii[1]
    if isinstance(obj, Rectangle):
        a, b, c, d = obj.bounds
        return (b - a) * (d - c)
    if isinstance(obj, Polygon):
        points = np.asarray(obj.vertices)
        return (
            abs(
                np.sum(
                    points[:, 0] * np.roll(points[:, 1], -1)
                    - points[:, 1] * np.roll(points[:, 0], -1)
                )
            )
            / 2
        )
    if isinstance(obj, PolygonWithHoles):
        return _object_area(obj.outer) - sum(_object_area(hole) for hole in obj.holes)
    theta = np.linspace(0, 2 * np.pi, 8193)
    return float(np.trapezoid(obj._radius(theta) ** 2, theta) / 2)


def _boundary_projection_coordinates(obj):
    """Project continuous boundary vertices or exact conic extrema onto x/y."""
    if isinstance(obj, Circle):
        return (
            (obj.center[0] - obj.radius, obj.center[0] + obj.radius),
            (obj.center[1] - obj.radius, obj.center[1] + obj.radius),
        )
    if isinstance(obj, Ellipse):
        rx, ry = obj.radii
        cosine, sine = np.cos(obj.angle), np.sin(obj.angle)
        half_x = np.hypot(rx * cosine, ry * sine)
        half_y = np.hypot(rx * sine, ry * cosine)
        return (
            (obj.center[0] - half_x, obj.center[0] + half_x),
            (obj.center[1] - half_y, obj.center[1] + half_y),
        )
    if isinstance(obj, Rectangle):
        a, b, c, d = obj.bounds
        points = np.asarray(((a, c), (a, d), (b, c), (b, d)))
    elif isinstance(obj, Polygon):
        points = np.asarray(obj.vertices)
    elif isinstance(obj, PolygonWithHoles):
        points = np.asarray([point for ring in obj.rings for point in ring.vertices])
    else:
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        radius = obj._radius(theta)
        points = np.column_stack(
            (
                obj.center[0] + radius * np.cos(theta + obj.angle),
                obj.center[1] + radius * np.sin(theta + obj.angle),
            )
        )
        a, b, c, d = obj.bounds
        points = np.concatenate((points, ((a, c), (a, d), (b, c), (b, d))))
    return tuple(tuple(float(value) for value in np.unique(points[:, axis])) for axis in (0, 1))


def scene_metrics(scene, *, domain=DOMAIN):
    objects = objects_from_scene(scene)
    bounds = np.asarray([obj.bounds for obj in objects], dtype=float)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 2] <= 0):
        raise ValueError("Every scene object must fit inside the domain")
    if np.any(bounds[:, 1] >= domain) or np.any(bounds[:, 3] >= domain):
        raise ValueError("Every scene object must fit inside the domain")

    # The first collection schema accepts only objects with a positive axis-aligned
    # bounding-box gap. This avoids ambiguous material precedence at intersections.
    minimum_bbox_gap = np.inf
    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            a, b, c, d = bounds[left]
            e, f, g, h = bounds[right]
            separated = b < e or f < a or d < g or h < c
            if not separated:
                raise ValueError("Collection objects require disjoint bounds and positive gaps")
            gap_x = max(0.0, e - b, a - f)
            gap_y = max(0.0, g - d, c - h)
            minimum_bbox_gap = min(minimum_bbox_gap, float(np.hypot(gap_x, gap_y)))

    x_intervals = _merge_projection_intervals(bounds[:, (0, 1)])
    y_intervals = _merge_projection_intervals(bounds[:, (2, 3)])
    merged_bounds = (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
    )
    areas = np.asarray([_object_area(obj) for obj in objects], dtype=float)
    boundary_coordinates = tuple(
        tuple(
            sorted(
                {
                    round(value, 10)
                    for obj in objects
                    for value in _boundary_projection_coordinates(obj)[axis]
                }
            )
        )
        for axis in (0, 1)
    )
    total_area = float(areas.sum())
    if total_area <= 0 or not np.isfinite(total_area):
        raise ValueError("Scene occupied area must be positive and finite")
    pec_areas = np.asarray([isinstance(obj.material, PEC) for obj in objects]) * areas
    dielectric_area = total_area - float(pec_areas.sum())
    eps_log = (
        sum(
            area * np.log(obj.material.epsilon_r)
            for area, obj in zip(areas, objects)
            if not isinstance(obj.material, PEC)
        )
        / dielectric_area
        if dielectric_area
        else 0.0
    )
    conductivity = (
        sum(
            area * obj.material.sigma_e
            for area, obj in zip(areas, objects)
            if not isinstance(obj.material, PEC)
        )
        / dielectric_area
        if dielectric_area
        else 0.0
    )
    definitions = scene.get("objects", [scene])
    feature_sizes = []
    for definition, obj in zip(definitions, objects):
        default_feature = min(obj.bounds[1] - obj.bounds[0], obj.bounds[3] - obj.bounds[2])
        feature_size = float(definition.get("feature_size_m", default_feature))
        if isinstance(obj, PolygonWithHoles):
            feature_size = min(feature_size, obj.minimum_feature_size)
        feature_sizes.append(feature_size)
    if any(value <= 0 or not np.isfinite(value) for value in feature_sizes):
        raise ValueError("Feature sizes must be positive and finite")
    return {
        "object_count": len(objects),
        "occupied_area_fraction": total_area / domain**2,
        "projected_x_support_fraction": sum(b - a for a, b in x_intervals) / domain,
        "projected_y_support_fraction": sum(b - a for a, b in y_intervals) / domain,
        "projected_x_extent_fraction": (merged_bounds[1] - merged_bounds[0]) / domain,
        "projected_y_extent_fraction": (merged_bounds[3] - merged_bounds[2]) / domain,
        "feature_size_m": min(feature_sizes),
        "bounds_m": merged_bounds,
        "projection_intervals": (x_intervals, y_intervals),
        "boundary_coordinates": boundary_coordinates,
        "minimum_axis_aligned_bbox_gap_m": (
            None if not np.isfinite(minimum_bbox_gap) else minimum_bbox_gap
        ),
        "pec_fraction": float(pec_areas.sum() / total_area),
        "epsilon_r_geometric_mean": float(np.exp(eps_log)),
        "sigma_e_area_mean": float(conductivity),
    }


def _merge_projection_intervals(intervals):
    merged = []
    for low, high in sorted((float(a), float(b)) for a, b in intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


def rasterize_v2(scene, resolution=512, *, domain=DOMAIN):
    """Nine model maps, with geometry evaluated continuously at subpixel points."""
    if resolution < 32 or int(resolution) != resolution:
        raise ValueError("Raster resolution must be an integer >=32")
    objects = objects_from_scene(scene)
    pixel = domain / resolution
    coordinate = (np.arange(resolution) + 0.5) * pixel
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    object_masks = [obj.contains(x, y) for obj in objects]
    interior = np.logical_or.reduce(object_masks)
    signed = (distance_transform_edt(interior) - distance_transform_edt(~interior)) * pixel
    fill = np.zeros_like(x, dtype=float)
    dielectric_fill = np.zeros_like(x, dtype=float)
    pec_fill = np.zeros_like(x, dtype=float)
    epsilon = np.zeros_like(x, dtype=float)
    sigma = np.zeros_like(x, dtype=float)
    for obj in objects:
        subpixel_fill = np.zeros_like(x, dtype=float)
        for ox in (-0.25, 0.25):
            for oy in (-0.25, 0.25):
                subpixel_fill += obj.contains(x + ox * pixel, y + oy * pixel) / 4
        fill += subpixel_fill
        if isinstance(obj.material, PEC):
            pec_fill += subpixel_fill
        else:
            dielectric_fill += subpixel_fill
            epsilon += subpixel_fill * (np.log(obj.material.epsilon_r) / np.log(30))
            sigma += subpixel_fill * (np.log1p(obj.material.sigma_e / 0.01) / np.log1p(0.35 / 0.01))
    metrics = scene_metrics(scene, domain=domain)
    feature = metrics["feature_size_m"]
    proximity = np.zeros_like(fill)
    if len(objects) > 1:
        distances = [distance_transform_edt(~mask) * pixel for mask in object_masks]
        proximity_scale = max(2 * pixel, 0.05 * domain)
        for first in range(len(objects)):
            for second in range(first + 1, len(objects)):
                proximity = np.maximum(
                    proximity,
                    np.exp(-(distances[first] + distances[second]) / proximity_scale),
                )
    channels = np.stack(
        (
            dielectric_fill,
            epsilon,
            sigma,
            pec_fill,
            np.clip(signed / max(0.25 * feature, pixel), -1, 1),
            np.exp(-abs(signed) / (2 * pixel)),
            proximity,
            x / domain,
            y / domain,
        )
    )
    return channels.astype(np.float32)


def conditioning_v2(
    scene, cells_x, cells_y, angle, frequencies=(0.8e9, 1e9, 1.2e9), *, domain=DOMAIN
):
    metrics = scene_metrics(scene, domain=domain)
    epsilon = metrics["epsilon_r_geometric_mean"]
    conductivity = metrics["sigma_e_area_mean"]
    result = np.array(
        (
            np.sin(angle),
            np.cos(angle),
            cells_x / 128,
            cells_y / 128,
            min(frequencies) / 1.2e9,
            max(frequencies) / 1.2e9,
            metrics["feature_size_m"] / domain,
            np.log(epsilon) / np.log(30),
            np.log1p(conductivity / 0.01) / np.log1p(0.35 / 0.01),
            1.0,  # maximum grading divided by three
            metrics["object_count"] / 10,
            metrics["pec_fraction"],
            metrics["occupied_area_fraction"],
            metrics["projected_x_support_fraction"],
            metrics["projected_y_support_fraction"],
        ),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("Conditioning must be finite")
    return result


def _rotate(points, center, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    return (np.asarray(points) @ rotation + center).tolist()


def _sample_geometry(family, index, rng, material):
    center = rng.uniform(0.43, 0.77, size=2)
    width = float(rng.uniform(0.17, 0.29))
    height = float(rng.uniform(0.15, 0.27))
    angle = float(rng.uniform(-0.75, 0.75))
    stage = "C2"
    common = {"material": material, "center_m": center.tolist()}
    if family == "circle":
        stage = "C0"
        return dict(common, shape="circle", radius_m=(width * height) ** 0.5 / 2), stage
    if family == "ellipse":
        stage = "C0" if index % 2 == 0 else "C1"
        return dict(
            common,
            shape="ellipse",
            radii_m=[width / 2, height / 2],
            angle_rad=angle if stage == "C1" else 0.0,
        ), stage
    if family == "rectangle":
        stage = "C0" if index % 2 == 0 else "C1"
        if stage == "C0":
            return dict(
                common,
                shape="rectangle",
                bounds_m=[
                    center[0] - width / 2,
                    center[0] + width / 2,
                    center[1] - height / 2,
                    center[1] + height / 2,
                ],
            ), stage
        return dict(
            common, shape="rotated_rectangle", width_m=width, height_m=height, angle_rad=angle
        ), stage
    if family == "triangle":
        local = [
            (-0.45 * width, -0.4 * height),
            (0.5 * width, -0.35 * height),
            (-0.05 * width, 0.5 * height),
        ]
    elif family == "convex_polygon":
        theta = np.arange(6) * 2 * np.pi / 6
        radius = rng.uniform(0.82, 1.0, 6)
        local = np.column_stack(
            (0.5 * width * radius * np.cos(theta), 0.5 * height * radius * np.sin(theta))
        )
    elif family == "concave_polygon":
        local = np.asarray(
            [
                (-0.5, -0.5),
                (0.5, -0.5),
                (0.5, 0.5),
                (0.15, 0.5),
                (0.15, 0.08),
                (-0.15, 0.08),
                (-0.15, 0.5),
                (-0.5, 0.5),
            ]
        ) * [width, height]
    elif family == "star":
        theta = np.arange(10) * np.pi / 5
        radius = np.where(np.arange(10) % 2 == 0, 0.5, 0.29)
        local = np.column_stack((width * radius * np.cos(theta), height * radius * np.sin(theta)))
    elif family == "smooth_lobed":
        theta = np.arange(16) * 2 * np.pi / 16
        base = min(width, height) / 2
        radii = base * (1 + 0.12 * np.cos(3 * theta) + 0.07 * np.sin(5 * theta))
        return dict(
            common, shape="smooth_lobed", radii_knots_m=radii.tolist(), angle_rad=angle
        ), stage
    else:
        raise ValueError(f"Unsupported family: {family}")
    return dict(common, shape=family, vertices_m=_rotate(local, center, angle)), stage


def _scaled_definition(definition, center, scale, material):
    """Scale one sampled shape and move it to a requested collection location."""
    result = {**definition, "material": material}
    origin = np.asarray(definition["center_m"], dtype=float)
    target = np.asarray(center, dtype=float)
    if "feature_size_m" in result:
        result["feature_size_m"] = float(result["feature_size_m"] * scale)

    def transform(points):
        return (target + scale * (np.asarray(points) - origin)).tolist()

    result["center_m"] = target.tolist()
    if definition["shape"] == "circle":
        result["radius_m"] = float(definition["radius_m"] * scale)
    elif definition["shape"] == "ellipse":
        result["radii_m"] = (np.asarray(definition["radii_m"]) * scale).tolist()
    elif definition["shape"] == "rectangle":
        low = target + scale * (np.asarray(definition["bounds_m"])[[0, 2]] - origin)
        high = target + scale * (np.asarray(definition["bounds_m"])[[1, 3]] - origin)
        result["bounds_m"] = [low[0], high[0], low[1], high[1]]
    elif definition["shape"] == "rotated_rectangle":
        result["width_m"] = float(definition["width_m"] * scale)
        result["height_m"] = float(definition["height_m"] * scale)
    elif definition["shape"] == "smooth_lobed":
        result["radii_knots_m"] = (np.asarray(definition["radii_knots_m"]) * scale).tolist()
    elif definition["shape"] == "polygon_with_holes":
        result["outer_ring_m"] = transform(definition["outer_ring_m"])
        result["holes_m"] = [transform(ring) for ring in definition["holes_m"]]
    else:
        result["vertices_m"] = transform(definition["vertices_m"])
    return result


def _engineering_outline(family, variant):
    """Return a simple normalized top-view silhouette outline."""
    if family not in ENGINEERING_FAMILIES or not 0 <= variant < 8:
        raise ValueError("Unknown engineering silhouette template")
    sweep = (-0.045, -0.015, 0.02, 0.05, -0.06, -0.03, 0.01, 0.045)[variant]
    if family == "aircraft":
        span = (0.39, 0.43, 0.46, 0.41, 0.35, 0.38, 0.44, 0.47)[variant]
        tail = (0.14, 0.17, 0.20, 0.16, 0.12, 0.15, 0.18, 0.21)[variant]
        return np.asarray(
            [
                (-0.50, 0.00),
                (-0.18, 0.055),
                (-0.08 + sweep, span),
                (0.01 + sweep, span),
                (0.025, 0.075),
                (0.31, 0.05),
                (0.40, tail),
                (0.48, tail * 0.92),
                (0.43, 0.015),
                (0.43, -0.015),
                (0.48, -tail * 0.92),
                (0.40, -tail),
                (0.31, -0.05),
                (0.025, -0.075),
                (0.01 + sweep, -span),
                (-0.08 + sweep, -span),
                (-0.18, -0.055),
            ]
        )
    if family == "ship":
        stern = (0.13, 0.16, 0.19, 0.145, 0.11, 0.14, 0.18, 0.20)[variant]
        beam = (0.17, 0.19, 0.165, 0.20, 0.15, 0.18, 0.21, 0.16)[variant]
        bow_x = (0.49, 0.50, 0.47, 0.50, 0.46, 0.48, 0.50, 0.45)[variant]
        return np.asarray(
            [
                (bow_x, 0.0),
                (0.34, 0.065),
                (0.02, 0.105),
                (-0.28, beam),
                (-0.45, stern),
                (-0.50, stern * 0.7),
                (-0.43, 0.04),
                (-0.43, -0.04),
                (-0.50, -stern * 0.7),
                (-0.45, -stern),
                (-0.28, -beam),
                (0.02, -0.105),
                (0.34, -0.065),
            ]
        )
    wheel = (0.22, 0.24, 0.20, 0.25, 0.18, 0.21, 0.23, 0.27)[variant]
    roof = (0.13, 0.15, 0.12, 0.14, 0.11, 0.14, 0.16, 0.12)[variant]
    return np.asarray(
        [
            (-0.50, -0.11),
            (-0.46, -0.17),
            (-0.33, -0.17),
            (-0.29, -0.105),
            (-0.22, -0.105),
            (-0.18, -0.17),
            (wheel, -0.17),
            (wheel + 0.04, -0.105),
            (0.34, -0.105),
            (0.43, -0.17),
            (0.49, -0.14),
            (0.50, 0.12),
            (0.45, 0.17),
            (0.32, 0.17),
            (0.26, roof),
            (-0.15, roof),
            (-0.25, 0.17),
            (-0.43, 0.17),
            (-0.50, 0.11),
        ]
    )


def _engineering_definition(family, variant, center, extent, angle, material, object_id=None):
    points = _engineering_outline(family, variant)
    points = points * (extent / np.ptp(points, axis=0).max())
    vertices = _rotate(points, np.asarray(center, dtype=float), angle)
    definition = {
        "shape": "concave_polygon",
        "vertices_m": vertices,
        "center_m": list(map(float, center)),
        "angle_rad": float(angle),
        "material": material,
        "feature_size_m": float(extent * (0.13 if family == "vehicle" else 0.16)),
        "template_id": f"{family}_outline_v{variant + 1}",
    }
    if object_id is not None:
        definition["object_id"] = object_id
    return definition


def generate_c8_c9_targets(*, silhouette_count=12, distributed_count=12, seed=20260923):
    """Build a frozen, held-out suite of simplified C8 silhouettes and C9 scenes.

    These deterministic procedural templates are top-view 2D outlines, not CAD
    assets. Every scene and geometry template carries explicit lineage metadata.
    """
    if silhouette_count < 3 or silhouette_count % 3:
        raise ValueError("C8 silhouette count must be a positive multiple of three")
    if distributed_count < 3:
        raise ValueError("At least three C9 distributed scenes are required")
    rng = np.random.default_rng(seed)
    material_choices = (
        {"kind": "dielectric", "epsilon_r": 2.0, "sigma_e_s_per_m": 0.0},
        {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0},
        {"kind": "pec"},
        {"kind": "dielectric", "epsilon_r": 2.0, "sigma_e_s_per_m": 0.01},
    )
    scenes = []
    for index in range(silhouette_count):
        family = ENGINEERING_FAMILIES[index % len(ENGINEERING_FAMILIES)]
        variant = (index // len(ENGINEERING_FAMILIES)) % 4
        material = material_choices[index % len(material_choices)]
        scene = {
            "schema_version": 3,
            "lineage_id": f"v3_c8_{family}_{variant + 1:02d}",
            "family": f"{family}_silhouette",
            "stage": "C8",
            "template_id": f"{family}_outline_v{variant + 1}",
            "split": "test",
            **_engineering_definition(
                family,
                variant,
                center=(0.6, 0.6),
                extent=float(rng.uniform(0.22, 0.27)),
                angle=float(rng.uniform(-0.65, 0.65)),
                material=material,
            ),
        }
        scene_metrics(scene)
        scenes.append(scene)

    slots = np.asarray(
        [
            (x, y)
            for y in (0.36, 0.52, 0.68, 0.84)
            for x in (0.36, 0.52, 0.68, 0.84)
            if np.hypot(x - DOMAIN / 2, y - DOMAIN / 2) >= 0.22
        ]
    )
    for index in range(distributed_count):
        scene_rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(f"{seed}:c9:{index}".encode()).digest()[:8], "little")
        )
        object_count = 2 + index % 5
        selected = scene_rng.choice(len(slots), size=object_count, replace=False)
        arrangement = ("compact", "aligned", "dispersed")[index % 3]
        members = []
        families = []
        for object_index, slot_index in enumerate(selected):
            family = ENGINEERING_FAMILIES[(index + object_index) % len(ENGINEERING_FAMILIES)]
            variant = (index + 2 * object_index) % 4
            material = material_choices[(index + object_index) % len(material_choices)]
            scale = {"compact": 0.11, "aligned": 0.125, "dispersed": 0.14}[arrangement]
            center = slots[int(slot_index)] + scene_rng.uniform(-0.004, 0.004, size=2)
            member = _engineering_definition(
                family,
                variant,
                center=center,
                extent=scale,
                angle=float(scene_rng.uniform(-0.7, 0.7)),
                material=material,
                object_id=f"object_{object_index:02d}",
            )
            members.append(member)
            families.append(family)
        scene = {
            "schema_version": 3,
            "lineage_id": f"v3_c9_{arrangement}_{index:03d}",
            "family": "distributed_engineering_scene",
            "stage": "C9",
            "arrangement": arrangement,
            "object_families": families,
            "objects": members,
            "split": "test",
        }
        scene_metrics(scene)
        scenes.append(scene)
    return scenes


def generate_development_scenes(seed=20260924):
    """Create eight development lineages with silhouette templates held out of C8."""
    foundation = generate_lineages(128, seed=seed)
    examples = []
    for index, (family, material_kind) in enumerate(
        (
            ("circle", "dielectric"),
            ("concave_polygon", "dielectric"),
            ("circle", "pec"),
        )
    ):
        scene = next(
            item
            for item in foundation
            if item["family"] == family and item["material"]["kind"] == material_kind
        )
        scene["lineage_id"] = f"dev_foundation_{index:02d}_{family}_{material_kind}"
        scene["split"] = "development"
        scene["development_role"] = "foundation"
        scene_metrics(scene)
        examples.append(scene)

    for index, family in enumerate(ENGINEERING_FAMILIES):
        material = {"kind": "dielectric", "epsilon_r": 2.0, "sigma_e_s_per_m": 0.0}
        scene = {
            "schema_version": 3,
            "lineage_id": f"dev_c8_{family}_template_v{index + 5}",
            "family": f"development_{family}_silhouette",
            "stage": "C8",
            "split": "development",
            "development_role": "heldout_silhouette_template",
            **_engineering_definition(
                family,
                index + 4,
                center=(0.6, 0.6),
                extent=0.24,
                angle=(-0.42, 0.17, 0.51)[index],
                material=material,
            ),
        }
        scene_metrics(scene)
        examples.append(scene)

    collections = generate_multiobject_lineages(24, seed=seed + 1)
    for stage in ("C3", "C5"):
        scene = next(row for row in collections if row["stage"] == stage)
        scene["lineage_id"] = f"dev_{stage.lower()}_scene_{seed}"
        scene["split"] = "development"
        scene["development_role"] = "separated_objects"
        scene_metrics(scene)
        examples.append(scene)

    if len(examples) != 8 or len({row["lineage_id"] for row in examples}) != 8:
        raise RuntimeError("The development set must contain eight independent lineages")
    return examples


def generate_compact_poc_scenes(seed=20260923):
    """Return 64 training, 16 validation, and 24 frozen C8/C9 target lineages."""
    foundation = generate_lineages(128, seed=seed)
    rng = np.random.default_rng(seed)
    selected_foundation = []
    for family in FAMILIES:
        family_rows = [
            row for row in foundation if row["family"] == family and row["split"] == "train"
        ]
        stages = sorted({row["stage"] for row in family_rows})
        if len(stages) == 2:
            # Preserve both axis-aligned and rotated primitive examples.
            for stage in stages:
                choices = [row for row in family_rows if row["stage"] == stage]
                material_rows = [row for row in choices if row["material"]["kind"] == "pec"]
                rng.shuffle(choices)
                if stage == stages[-1] and material_rows:
                    picked = material_rows[0]
                else:
                    picked = choices[0]
                selected_foundation.append(picked)
            remainder = [row for row in family_rows if row not in selected_foundation]
            rng.shuffle(remainder)
            selected_foundation.extend(remainder[:4])
        else:
            pec_rows = [row for row in family_rows if row["material"]["kind"] == "pec"]
            dielectric_rows = [row for row in family_rows if row["material"]["kind"] != "pec"]
            rng.shuffle(dielectric_rows)
            if pec_rows:
                selected_foundation.append(pec_rows[0])
            selected_foundation.extend(dielectric_rows[: 5 if pec_rows else 6])

    if len(selected_foundation) != 48:
        raise RuntimeError("Compact foundation selection must contain 48 lineages")

    foundation_validation = [row for row in foundation if row["split"] == "validation"]
    selected_validation = []
    for family in FAMILIES:
        rows = [row for row in foundation_validation if row["family"] == family]
        if not rows:
            raise RuntimeError(f"No validation lineage for {family}")
        selected_validation.append(rows[seed % len(rows)])
    remainder = [row for row in foundation_validation if row not in selected_validation]
    rng.shuffle(remainder)
    selected_validation.extend(remainder[:5])
    if len(selected_validation) != 13:
        raise RuntimeError("Compact foundation validation selection must contain 13 lineages")

    collections = generate_multiobject_lineages(24, seed=seed)
    collection_train = [row for row in collections if row["split"] == "train"]
    selected_collections = [row for row in collection_train if row["stage"] == "C3"]
    c5_rows = [row for row in collection_train if row["stage"] == "C5"]
    rng.shuffle(c5_rows)
    selected_collections.extend(c5_rows[: 16 - len(selected_collections)])
    collection_validation = [row for row in collections if row["split"] == "validation"]
    targets = generate_c8_c9_targets(seed=seed)

    for row in selected_foundation + selected_collections:
        row["split"] = "train"
    for row in selected_validation + collection_validation:
        row["split"] = "validation"
    scenes = (
        selected_foundation
        + selected_collections
        + selected_validation
        + collection_validation
        + targets
    )
    counts = {
        split: sum(row["split"] == split for row in scenes)
        for split in ("train", "validation", "test")
    }
    if counts != {"train": 64, "validation": 16, "test": 24}:
        raise RuntimeError(f"Unexpected compact curriculum split sizes: {counts}")
    identifiers = [row["lineage_id"] for row in scenes]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Compact curriculum lineages must be unique")
    return scenes


def generate_c4_gap_sweep(
    gaps_m=(0.002, 0.005, 0.01, 0.02),
    *,
    radius_m=0.06,
    epsilon_r=4.0,
    center_m=(0.6, 0.6),
    frequency_hz=1e9,
    split="development",
):
    """Create deterministic two-cylinder C4 scenes with controlled positive gaps.

    These cases describe gap geometry and representation stress; the returned
    scenes are not qualified FDTD labels. Keep the initial sweep dielectric so
    PEC split-edge handling does not confound the gap experiment.
    """
    gaps = tuple(float(gap) for gap in gaps_m)
    center = np.asarray(center_m, dtype=float)
    if (
        not gaps
        or any(not np.isfinite(gap) or gap <= 0 for gap in gaps)
        or not np.isfinite(radius_m)
        or radius_m <= 0
        or not np.isfinite(epsilon_r)
        or epsilon_r <= 1
        or center.shape != (2,)
        or not np.isfinite(center).all()
        or not np.isfinite(frequency_hz)
        or frequency_hz <= 0
        or split not in {"development", "train", "validation", "test"}
    ):
        raise ValueError("C4 gaps, radius, material, center, frequency, and split must be valid")
    material = {"kind": "dielectric", "epsilon_r": float(epsilon_r), "sigma_e_s_per_m": 0.0}
    wavelength = C0 / frequency_hz
    rows = []
    for index, gap in enumerate(gaps):
        offset = radius_m + gap / 2
        objects = [
            {
                "object_id": f"gap_pair_{index:02d}_{side}",
                "shape": "circle",
                "center_m": [float(center[0] + sign * offset), float(center[1])],
                "radius_m": float(radius_m),
                "feature_size_m": float(gap),
                "material": material.copy(),
            }
            for side, sign in (("left", -1), ("right", 1))
        ]
        scene = {
            "schema_version": 3,
            "lineage_id": f"v3_c4_gap_{index:03d}",
            "family": "two_circle_gap_sweep",
            "stage": "C4",
            "split": split,
            "objects": objects,
            "gap_m": gap,
            "gap_wavelengths": gap / wavelength,
            "center_separation_m": 2 * radius_m + gap,
            "feature_size_m": gap,
            "frequency_hz": float(frequency_hz),
            "geometry_qualification": "continuous_positive_gap_only",
        }
        scene_metrics(scene)
        rows.append(scene)
    return rows


def c4_gap_sampling_diagnostics(scene, cells, *, raster_resolution=512, material_samples=12):
    """Quantify grid/raster sampling of a C4 gap without claiming it is resolved."""
    if scene.get("stage") != "C4" or "gap_m" not in scene:
        raise ValueError("C4 gap diagnostics require a generated C4 scene")
    if isinstance(cells, bool) or int(cells) != cells or cells <= 0:
        raise ValueError("Cell count must be a positive integer")
    if isinstance(raster_resolution, bool) or int(raster_resolution) != raster_resolution:
        raise ValueError("Raster resolution must be an integer")
    if isinstance(material_samples, bool) or int(material_samples) != material_samples:
        raise ValueError("Material samples must be an integer")
    if raster_resolution <= 0 or material_samples <= 0:
        raise ValueError("Raster resolution and material samples must be positive")
    gap = float(scene["gap_m"])
    cell_width = DOMAIN / int(cells)
    raster_pixel = DOMAIN / int(raster_resolution)
    quadrature_spacing = cell_width / int(material_samples)
    return {
        "gap_m": gap,
        "uniform_cell_width_m": cell_width,
        "gap_in_uniform_cells": gap / cell_width,
        "gap_in_raster_pixels": gap / raster_pixel,
        "nominal_quadrature_samples_across_gap": gap / quadrature_spacing,
        "subcell_gap": gap < cell_width,
        "subpixel_gap": gap < raster_pixel,
        "qualification_status": "requires_convergence_study",
    }


def generate_multiobject_lineages(
    count,
    *,
    seed=20260923,
    object_counts=(2, 3, 4, 5),
    mixed_materials=False,
):
    """Generate reproducible, dielectric C3/C5 scenes with separated objects.

    This first collection generator places compact objects on a jittered 4x4
    lattice. It intentionally reserves the source neighborhood and emits
    positive axis-aligned bounding-box gaps for unambiguous material sampling.
    Pass ``object_counts=range(2, 11)`` to extend C5 lineages through ten
    objects. ``mixed_materials=True`` assigns each dielectric object epsilon_r
    2 or 4 while the default preserves homogeneous scene materials.
    """
    if isinstance(count, bool) or int(count) != count or count < 12:
        raise ValueError("At least twelve integer collection lineages are required")
    object_counts = tuple(object_counts)
    if (
        not object_counts
        or len(set(object_counts)) != len(object_counts)
        or any(
            isinstance(value, bool) or int(value) != value or not 2 <= value <= 10
            for value in object_counts
        )
        or 2 not in object_counts
        or not any(value > 2 for value in object_counts)
    ):
        raise ValueError("object_counts must uniquely include two and one count from 3 through 10")
    if not isinstance(mixed_materials, bool):
        raise ValueError("mixed_materials must be boolean")
    count = int(count)
    positions = np.array([0.34, 0.507, 0.693, 0.86])
    slots = [
        (x, y)
        for y in positions
        for x in positions
        if np.hypot(x - DOMAIN / 2, y - DOMAIN / 2) >= 0.2
    ]
    rows = []
    for index in range(count):
        digest = hashlib.sha256(f"{seed}:collection:{index}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        object_count = object_counts[index % len(object_counts)]
        selected = rng.choice(len(slots), size=object_count, replace=False)
        material_epsilon = float((2, 4)[index % 2])
        object_definitions = []
        object_families = []
        for object_index, slot_index in enumerate(selected):
            object_material = {
                "kind": "dielectric",
                "epsilon_r": float((2, 4)[(index + object_index) % 2])
                if mixed_materials
                else material_epsilon,
                "sigma_e_s_per_m": 0.0,
            }
            family = FAMILIES[(index + object_index * 3) % len(FAMILIES)]
            definition, _ = _sample_geometry(
                family,
                index * 11 + object_index,
                rng,
                object_material,
            )
            center = np.asarray(slots[int(slot_index)]) + rng.uniform(-0.012, 0.012, size=2)
            obj = _scaled_definition(definition, center, 0.42, definition["material"])
            obj["object_id"] = f"object_{object_index:02d}"
            object_definitions.append(obj)
            object_families.append(family)
        stage = "C3" if object_count == 2 else "C5"
        scene = {
            "schema_version": 3,
            "lineage_id": f"v3_multi_{index:04d}",
            "family": "multi_object",
            "stage": stage,
            "object_families": object_families,
            "objects": object_definitions,
            "split": "train",
        }
        if mixed_materials:
            scene["material_assignment"] = "per_object_epsilon_2_or_4"
        scene_metrics(scene)
        rows.append(scene)

    # Split complete scene lineages by stage, so each stage is represented in
    # train/validation/test when the requested corpus contains enough examples.
    for stage in ("C3", "C5"):
        indices = [i for i, row in enumerate(rows) if row["stage"] == stage]
        order = np.random.default_rng(seed + len(indices) + int(stage == "C5")).permutation(indices)
        held_out = max(1, round(len(order) / 8))
        if len(order) < 3:
            raise ValueError("Collection corpus needs at least three lineages per stage")
        for rank, row_index in enumerate(order):
            rows[row_index]["split"] = (
                "test" if rank < held_out else ("validation" if rank < 2 * held_out else "train")
            )
    return rows


def generate_lineages(count, *, seed=20260923):
    """A deterministic 75/25 dielectric/PEC corpus grouped by shape lineage."""
    if count not in (128, 256):
        raise ValueError("The first pilot supports 128 or 256 independent lineages")
    per_family = count // len(FAMILIES)
    rows = []
    for family in FAMILIES:
        candidates = []
        for index in range(per_family):
            digest = hashlib.sha256(f"{seed}:{family}:{index}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            pec = index % 4 == 3
            material = (
                {"kind": "pec"}
                if pec
                else {
                    "kind": "dielectric",
                    "epsilon_r": (2, 4, 8, 12)[(index // 4) % 4],
                    "sigma_e_s_per_m": 0.01 if index % 2 else 0.0,
                }
            )
            geometry, stage = _sample_geometry(family, index, rng, material)
            scene = {
                "schema_version": 2,
                "lineage_id": f"v2_{family}_{index:03d}",
                "family": family,
                "stage": stage,
                **geometry,
            }
            metrics = scene_metrics(scene)
            if (
                not 0.005 <= metrics["occupied_area_fraction"] <= 0.08
                or max(
                    metrics["projected_x_support_fraction"], metrics["projected_y_support_fraction"]
                )
                > 0.35
            ):
                raise ValueError(
                    f"Generated scene exceeds pilot geometry bounds: {scene['lineage_id']}"
                )
            candidates.append(scene)
        for pec in (False, True):
            group = [scene for scene in candidates if (scene["material"]["kind"] == "pec") == pec]
            order = np.random.default_rng(seed + len(rows) + int(pec)).permutation(len(group))
            test_and_val = max(1, round(len(group) / 8)) if len(group) > 12 else 1
            for rank, item in enumerate(order):
                scene = group[item]
                scene["split"] = (
                    "test"
                    if rank < test_and_val
                    else ("validation" if rank < 2 * test_and_val else "train")
                )
                rows.append(scene)
    return rows


def scene_digest(scene):
    return hashlib.sha256(
        json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
