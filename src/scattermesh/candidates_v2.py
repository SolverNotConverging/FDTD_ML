"""Bounded exact-budget geometry-aware mesh policies for curriculum scenes."""

import hashlib

import numpy as np

from .curriculum_v2 import DOMAIN, scene_metrics
from .profiles_v2 import axis_probability, probability_axis

CANDIDATE_NAMES = (
    "uniform",
    "center",
    "interface",
    "wide",
    "hybrid",
    "smooth_0",
    "smooth_1",
    "smooth_2",
    "smooth_3",
    "refine_0",
    "refine_1",
    "refine_2",
)


def _seed(*parts):
    encoded = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _bump(coordinate, center, width):
    return np.exp(-0.5 * ((coordinate - center) / width) ** 2)


def candidate_axes(scene, cells, name, *, selected_axes=None, domain=DOMAIN):
    """Return two legal axes without moving or anchoring the physical geometry."""
    if name == "learned":
        if selected_axes is None:
            raise ValueError("Learned policy requires projected x/y axes")
        x, y = (np.asarray(axis, dtype=float) for axis in selected_axes)
        if any(
            len(axis) != cells + 1
            or not np.isclose(axis[0], 0)
            or not np.isclose(axis[-1], domain)
            or np.any(np.diff(axis) <= 0)
            for axis in (x, y)
        ):
            raise ValueError("Learned axes must be increasing and meet exact cell count")
        return x, y, (0.0, 0.0)
    if name not in CANDIDATE_NAMES or cells not in (32, 48, 64, 96, 128, 192, 256, 384, 512):
        raise ValueError("Unknown candidate policy or unsupported cell budget")
    if name == "uniform":
        axis = np.linspace(0, domain, cells + 1)
        return axis.copy(), axis.copy(), (0.0, 0.0)
    metrics = scene_metrics(scene, domain=domain)
    a, b, c, d = metrics["bounds_m"]
    projection_intervals = metrics["projection_intervals"]
    boundary_coordinates = metrics["boundary_coordinates"]
    coordinate = (np.arange(512) + 0.5) * domain / 512
    results, repairs = [], []
    for axis, intervals in enumerate(projection_intervals):
        low, high = (a, b) if axis == 0 else (c, d)
        center = (low + high) / 2
        extent = high - low
        fine = max(0.12 * extent, 1.5 * domain / 512)
        wide = max(0.55 * extent, fine)
        rho = np.ones_like(coordinate)
        if name in {"center", "hybrid"}:
            for left, right in intervals:
                rho += 1.4 * _bump(coordinate, (left + right) / 2, max(0.35 * (right - left), fine))
        if name in {"interface", "hybrid"}:
            local_width = max(0.15 * min(metrics["feature_size_m"], extent), 1.5 * domain / 512)
            for boundary in boundary_coordinates[axis]:
                rho += 1.1 * _bump(coordinate, boundary, local_width)
        if name == "wide":
            rho += 1.0 * _bump(coordinate, center, wide)
        if name.startswith("smooth_"):
            rng = np.random.default_rng(_seed(scene["lineage_id"], cells, name, axis))
            for _ in range(3):
                left, right = intervals[int(rng.integers(len(intervals)))]
                focal = rng.uniform(left, right)
                width = max((right - left) * rng.uniform(0.12, 0.75), fine)
                rho += rng.uniform(0.25, 2.0) * _bump(coordinate, focal, width)
        if name.startswith("refine_"):
            if selected_axes is None:
                raise ValueError("Local refinement requires a selected feasible candidate")
            rho = axis_probability(selected_axes[axis], 512, length=domain) * 512
            rng = np.random.default_rng(_seed(scene["lineage_id"], cells, name, axis))
            left, right = intervals[int(rng.integers(len(intervals)))]
            focal = rng.uniform(left, right)
            width = max((right - left) * rng.uniform(0.1, 0.4), fine)
            rho *= 1 + 0.35 * _bump(coordinate, focal, width)
        nodes, repair = probability_axis(rho, cells, length=domain, max_ratio=3.0)
        results.append(nodes)
        repairs.append(repair)
    return results[0], results[1], tuple(repairs)
