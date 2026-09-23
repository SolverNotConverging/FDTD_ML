"""Versioned C0--C2 scene definitions and continuous-geometry model inputs."""

import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import PEC, Circle, Material, Rectangle
from .geometry_v2 import Ellipse, Polygon, SmoothLobed, oriented_rectangle

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


def _material(definition):
    if definition["kind"] == "pec":
        return PEC()
    if definition["kind"] != "dielectric":
        raise ValueError("Material kind must be dielectric or pec")
    return Material(float(definition["epsilon_r"]), float(definition.get("sigma_e_s_per_m", 0)))


def object_from_scene(scene):
    """Construct the continuous object used by both solver and rasterizer."""
    material = _material(scene["material"])
    shape = scene["shape"]
    if shape == "circle":
        return Circle(tuple(scene["center_m"]), float(scene["radius_m"]), material)
    if shape == "ellipse":
        return Ellipse(
            tuple(scene["center_m"]), tuple(scene["radii_m"]), float(scene["angle_rad"]), material
        )
    if shape == "rectangle":
        return Rectangle(tuple(scene["bounds_m"]), material)
    if shape == "rotated_rectangle":
        return oriented_rectangle(
            scene["center_m"], scene["width_m"], scene["height_m"], scene["angle_rad"], material
        )
    if shape in {"triangle", "convex_polygon", "concave_polygon", "star"}:
        return Polygon(tuple(tuple(point) for point in scene["vertices_m"]), material)
    if shape == "smooth_lobed":
        return SmoothLobed(
            tuple(scene["center_m"]),
            tuple(scene["radii_knots_m"]),
            float(scene["angle_rad"]),
            material,
        )
    raise ValueError(f"Unsupported v2 shape: {shape}")


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
    theta = np.linspace(0, 2 * np.pi, 8193)
    return float(np.trapezoid(obj._radius(theta) ** 2, theta) / 2)


def scene_metrics(scene, *, domain=DOMAIN):
    obj = object_from_scene(scene)
    a, b, c, d = obj.bounds
    if min(a, c) <= 0 or max(b, d) >= domain:
        raise ValueError("Scene object must fit in the domain")
    feature = float(scene.get("feature_size_m", min(b - a, d - c)))
    if feature <= 0 or not np.isfinite(feature):
        raise ValueError("Feature size must be positive")
    return {
        "occupied_area_fraction": float(_object_area(obj) / domain**2),
        "projected_x_support_fraction": float((b - a) / domain),
        "projected_y_support_fraction": float((d - c) / domain),
        "feature_size_m": feature,
        "bounds_m": (float(a), float(b), float(c), float(d)),
        "pec_fraction": float(isinstance(obj.material, PEC)),
    }


def rasterize_v2(scene, resolution=512, *, domain=DOMAIN):
    """Nine model maps, with geometry evaluated continuously at subpixel points."""
    if resolution < 32 or int(resolution) != resolution:
        raise ValueError("Raster resolution must be an integer >=32")
    obj = object_from_scene(scene)
    pixel = domain / resolution
    coordinate = (np.arange(resolution) + 0.5) * pixel
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    interior = obj.contains(x, y)
    signed = (distance_transform_edt(interior) - distance_transform_edt(~interior)) * pixel
    fill = np.zeros_like(x)
    for ox in (-0.25, 0.25):
        for oy in (-0.25, 0.25):
            fill += obj.contains(x + ox * pixel, y + oy * pixel) / 4
    metrics = scene_metrics(scene, domain=domain)
    feature = metrics["feature_size_m"]
    dielectric = not isinstance(obj.material, PEC)
    dielectric_fill = fill if dielectric else np.zeros_like(fill)
    pec_fill = fill if not dielectric else np.zeros_like(fill)
    epsilon = dielectric_fill * (np.log(obj.material.epsilon_r) / np.log(30) if dielectric else 0)
    sigma = dielectric_fill * (
        np.log1p(obj.material.sigma_e / 0.01) / np.log1p(0.35 / 0.01) if dielectric else 0
    )
    channels = np.stack(
        (
            dielectric_fill,
            epsilon,
            sigma,
            pec_fill,
            np.clip(signed / max(0.25 * feature, pixel), -1, 1),
            np.exp(-abs(signed) / (2 * pixel)),
            np.zeros_like(fill),  # pair proximity is zero for the single-object pilot
            x / domain,
            y / domain,
        )
    )
    return channels.astype(np.float32)


def conditioning_v2(
    scene, cells_x, cells_y, angle, frequencies=(0.8e9, 1e9, 1.2e9), *, domain=DOMAIN
):
    metrics = scene_metrics(scene, domain=domain)
    material = scene["material"]
    epsilon = float(material.get("epsilon_r", 1))
    conductivity = float(material.get("sigma_e_s_per_m", 0))
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
            0.25,  # one object divided by four
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
