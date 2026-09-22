#!/usr/bin/env python3
"""Build a deterministic, preflighted sparse-pair campaign from qualified policies."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scattermesh import Grid, sparse_candidate_axes, sparse_cluster_metrics, sparse_scene_objects
from scattermesh.conformal import CutCellPEC
from scattermesh.constants import C0, EPS0
from scattermesh.generalization import widest_non_pml_monitor_bounds
from scattermesh.geometry import PEC

ROOT = Path(__file__).resolve().parents[1]


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _material(rng, pec=False):
    if pec:
        return {"kind": "pec"}
    epsilon = float(rng.choice([2, 3, 4, 6, 8, 12, 20, 30]))
    if epsilon >= 12:
        tangent = float(rng.choice([0.10, 0.15, 0.20]))
    else:
        tangent = float(rng.choice([0.0, 0.03, 0.08, 0.15]))
    sigma = tangent * 2 * np.pi * 1e9 * EPS0 * epsilon
    return {
        "kind": "dielectric",
        "epsilon_r": epsilon,
        "sigma_e_s_per_m": float(sigma),
        "loss_tangent_at_1ghz": tangent,
    }


def _shape(rng, kind, center, material):
    if kind == "circle":
        radius = float(rng.uniform(0.035, 0.075))
        return {
            "shape": "circle",
            "center_m": [float(center[0]), float(center[1])],
            "radius_m": radius,
            "material": material,
        }, np.array([radius, radius])
    width = float(rng.uniform(0.07, 0.15))
    height = float(rng.uniform(0.07, 0.15))
    return {
        "shape": "rectangle",
        "bounds_m": [
            float(center[0] - width / 2),
            float(center[0] + width / 2),
            float(center[1] - height / 2),
            float(center[1] + height / 2),
        ],
        "material": material,
    }, np.array([width / 2, height / 2])


def _move(definition, center):
    updated = dict(definition)
    if definition["shape"] == "circle":
        updated["center_m"] = [float(center[0]), float(center[1])]
    else:
        a, b, c, d = definition["bounds_m"]
        half_x, half_y = (b - a) / 2, (d - c) / 2
        updated["bounds_m"] = [
            float(center[0] - half_x),
            float(center[0] + half_x),
            float(center[1] - half_y),
            float(center[1] + half_y),
        ]
    return updated


def _make_scene(rng, family, serial):
    if family == "dd_circles":
        kinds, pec = ("circle", "circle"), (False, False)
    elif family == "dd_mixed_shapes":
        kinds = ("circle", "rectangle") if serial % 2 == 0 else ("rectangle", "rectangle")
        pec = (False, False)
    elif family == "dielectric_pec":
        kinds = ("circle", "rectangle") if serial % 2 == 0 else ("rectangle", "rectangle")
        pec = (False, True)
    elif family == "pec_rectangles":
        kinds, pec = ("rectangle", "rectangle"), (True, True)
    else:
        raise ValueError(f"Unsupported family: {family}")
    center = rng.uniform(0.40, 0.80, size=2)
    first, half_first = _shape(rng, kinds[0], center, _material(rng, pec[0]))
    second, half_second = _shape(rng, kinds[1], center, _material(rng, pec[1]))
    gap = float(np.exp(rng.uniform(np.log(0.012), np.log(0.070))))
    orientation = float(rng.choice([0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]))
    direction = np.array([math.cos(orientation), math.sin(orientation)])
    # Bounding-radius separation is conservative for rotated circle/rectangle pairs.
    extent_first = float(np.sum(np.abs(direction) * half_first))
    extent_second = float(np.sum(np.abs(direction) * half_second))
    separation = extent_first + extent_second + gap
    first_center = center - direction * separation / 2
    second_center = center + direction * separation / 2
    first, second = _move(first, first_center), _move(second, second_center)
    definition = {
        "scene_id": f"sparse_{family}_{serial:03d}_{_hash([first, second])[:8]}",
        "family_id": family,
        "gap_stratum": "close" if gap < 0.024 else "moderate" if gap < 0.045 else "wide",
        "incidence_angle_rad": float(rng.uniform(0, 2 * np.pi)),
        "objects": [first, second],
    }
    return definition


def _preflight(scene, config, *, minimum_pec_cfl_fraction):
    metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
    if not (
        0.005 <= metrics["occupied_area_fraction"] <= 0.12
        and metrics["projected_x_support_fraction"] <= 0.45
        and metrics["projected_y_support_fraction"] <= 0.45
    ):
        return False
    objects = sparse_scene_objects(scene)
    pec = tuple(obj for obj in objects if isinstance(obj.material, PEC))
    for cells in (*config["budgets"], *config["reference_budgets"]):
        policies = (
            config["policies"]
            if cells in config["budgets"]
            else ({"kind": "uniform", "max_ratio": 1.0},)
        )
        for policy in policies:
            try:
                x, y = sparse_candidate_axes(config["domain_m"], cells, scene, policy)
                grid = Grid(x, y, max_ratio=policy["max_ratio"])
                widest_non_pml_monitor_bounds(
                    grid, [obj.bounds for obj in objects], config["pml_thickness_m"]
                )
                if pec:
                    cut = CutCellPEC(grid, pec, config["pec_mode"])
                    dx, dy = np.diff(x), np.diff(y)
                    grid_dt = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
                    if cut.stable_time_step() / grid_dt < minimum_pec_cfl_fraction:
                        return False
            except ValueError:
                return False
    return True


def build(template, output, *, count_per_family=24, seed=20260923, max_attempts=200000):
    template = Path(template)
    config = json.loads(template.read_text())
    if count_per_family < 8 or count_per_family % 8:
        raise ValueError("count per family must be a positive multiple of eight")
    rng = np.random.default_rng(seed)
    families = ("dd_circles", "dd_mixed_shapes", "dielectric_pec", "pec_rectangles")
    scenes = []
    attempts = Counter()
    for family in families:
        accepted = []
        while len(accepted) < count_per_family:
            attempts[family] += 1
            if sum(attempts.values()) > max_attempts:
                raise RuntimeError(f"Could not generate enough feasible sparse pairs: {attempts}")
            scene = _make_scene(rng, family, len(accepted))
            if _preflight(scene, config, minimum_pec_cfl_fraction=0.08):
                accepted.append(scene)
        order = rng.permutation(len(accepted))
        train_count = 3 * count_per_family // 4
        validation_count = count_per_family // 8
        for rank, index in enumerate(order):
            accepted[index]["split"] = (
                "train"
                if rank < train_count
                else "validation"
                if rank < train_count + validation_count
                else "test"
            )
        scenes.extend(accepted)
    config["purpose"] = "scene-diverse sparse pair campaign after qualified M4.3A/B pilots"
    config["generator"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "seed": seed,
        "count_per_family": count_per_family,
        "attempts_by_family": dict(attempts),
    }
    config["scenes"] = scenes
    config["campaign_id"] = "sparse_pairs_" + _hash(
        {"generator": config["generator"], "scenes": scenes}
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/sparse_mixed_pair_headroom_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--count-per-family", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    config = build(args.template, args.output, count_per_family=args.count_per_family, seed=args.seed)
    print(
        json.dumps(
            {
                "campaign_id": config["campaign_id"],
                "scene_count": len(config["scenes"]),
                "split_counts": dict(Counter(scene["split"] for scene in config["scenes"])),
                "attempts_by_family": config["generator"]["attempts_by_family"],
                "reference_cases": len(config["scenes"]) * len(config["reference_budgets"]),
                "candidate_cases": len(config["scenes"]) * len(config["budgets"]) * len(config["policies"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
