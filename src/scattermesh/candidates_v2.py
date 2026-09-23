"""Bounded exact-budget teacher mesh policies for the C0--C2 pilot."""

import hashlib

import numpy as np

from .curriculum_v2 import DOMAIN, scene_metrics
from .distillation import axis_probability, probability_axis

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
    coordinate = (np.arange(512) + 0.5) * domain / 512
    results, repairs = [], []
    for axis, (low, high) in enumerate(((a, b), (c, d))):
        center = (low + high) / 2
        extent = high - low
        fine = max(0.12 * extent, 1.5 * domain / 512)
        wide = max(0.55 * extent, fine)
        rho = np.ones_like(coordinate)
        if name in {"center", "hybrid"}:
            rho += 1.4 * _bump(coordinate, center, 0.35 * extent)
        if name in {"interface", "hybrid"}:
            rho += 1.1 * (_bump(coordinate, low, fine) + _bump(coordinate, high, fine))
        if name == "wide":
            rho += 1.0 * _bump(coordinate, center, wide)
        if name.startswith("smooth_"):
            rng = np.random.default_rng(_seed(scene["lineage_id"], cells, name, axis))
            for _ in range(3):
                focal = center + rng.uniform(-0.65, 0.65) * extent
                width = extent * rng.uniform(0.12, 0.75)
                rho += rng.uniform(0.25, 2.0) * _bump(coordinate, focal, width)
        if name.startswith("refine_"):
            if selected_axes is None:
                raise ValueError("Local refinement requires a selected feasible candidate")
            rho = axis_probability(selected_axes[axis], 512, length=domain) * 512
            rng = np.random.default_rng(_seed(scene["lineage_id"], cells, name, axis))
            focal = center + rng.uniform(-0.5, 0.5) * extent
            width = extent * rng.uniform(0.1, 0.4)
            rho *= 1 + 0.35 * _bump(coordinate, focal, width)
        nodes, repair = probability_axis(rho, cells, length=domain, max_ratio=3.0)
        results.append(nodes)
        repairs.append(repair)
    return results[0], results[1], tuple(repairs)
