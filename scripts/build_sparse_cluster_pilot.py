#!/usr/bin/env python3
"""Generate a localized three/four-object scattering pilot with multiple gaps."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from build_sparse_pair_campaign import _material, _move, _preflight, _shape

from scattermesh.sparse import sparse_cluster_metrics

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "three_dielectric": (False, False, False),
    "three_mixed": (False, False, True),
    "four_dielectric": (False, False, False, False),
    "four_mixed": (False, True, False, True),
    "three_mixed_circle_pec": (False, False, True),
    "four_mixed_circle_pec": (False, True, False, True),
}
GAP_RANGES = ((0.012, 0.022), (0.026, 0.042), (0.050, 0.070))


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _scene(rng, family, serial, domain):
    pec_flags = FAMILIES[family]
    shapes = [
        "circle" if pec and family.endswith("circle_pec") else
        "rectangle" if pec or (serial + index) % 2 else "circle"
        for index, pec in enumerate(pec_flags)
    ]
    definitions, halves = [], []
    for shape, pec in zip(shapes, pec_flags):
        if pec and shape == "circle":
            radius = float(rng.uniform(0.055, 0.075))
            definition = {
                "shape": "circle", "center_m": [0.0, 0.0], "radius_m": radius,
                "material": {"kind": "pec"},
            }
            half = np.array([radius, radius])
        else:
            definition, half = _shape(rng, shape, np.zeros(2), _material(rng, pec))
        definitions.append(definition)
        halves.append(half)
    low, high = GAP_RANGES[serial % len(GAP_RANGES)]
    primary_gap = float(rng.uniform(low, high))
    wide_gap = float(rng.uniform(*GAP_RANGES[-1]))
    gap_x, gap_y = (
        (primary_gap, wide_gap) if serial % 2 else (wide_gap, primary_gap)
    )
    centers = [np.zeros(2)]
    centers.append(np.array([halves[0][0] + halves[1][0] + gap_x, 0.0]))
    centers.append(np.array([0.0, halves[0][1] + halves[2][1] + gap_y]))
    if len(definitions) == 4:
        centers.append(np.array([
            centers[1][0], centers[1][1] + halves[1][1] + halves[3][1] + gap_y,
        ]))
    rotation = int(rng.integers(0, 4))
    for _ in range(rotation):
        centers = [np.array([-center[1], center[0]]) for center in centers]
    centers_array = np.stack(centers)
    center_of_bounds = (centers_array.min(axis=0) + centers_array.max(axis=0)) / 2
    destination = rng.uniform(0.43, domain - 0.43, size=2)
    objects = [
        _move(definition, center - center_of_bounds + destination)
        for definition, center in zip(definitions, centers)
    ]
    scene = {
        "scene_id": f"sparse_{family}_{serial:03d}_{_digest(objects)[:8]}",
        "family_id": family,
        "incidence_angle_rad": float(rng.uniform(0, 2 * np.pi)),
        "nominal_gaps_m": [gap_x, gap_y],
        "objects": objects,
    }
    metrics = sparse_cluster_metrics(scene, domain=domain)
    gap = metrics["minimum_gap_m"]
    scene["gap_stratum"] = "close" if gap < 0.024 else "moderate" if gap < 0.045 else "wide"
    return scene


def build(template, output, *, count_per_family=8, seed=20260924, max_attempts=200000):
    template, output = Path(template), Path(output)
    if count_per_family < 8 or count_per_family % 8:
        raise ValueError("count per family must be a positive multiple of eight")
    config = json.loads(template.read_text())
    rng = np.random.default_rng(seed)
    scenes = []
    attempts = Counter()
    held_out_gaps = (
        ("close", "moderate"),
        ("moderate", "wide"),
        ("wide", "close"),
        ("close", "wide"),
        ("moderate", "close"),
        ("wide", "moderate"),
    )
    for family_index, family in enumerate(FAMILIES):
        accepted = []
        while len(accepted) < count_per_family:
            attempts[family] += 1
            if sum(attempts.values()) > max_attempts:
                raise RuntimeError(f"Could not generate enough feasible clusters: {attempts}")
            try:
                scene = _scene(rng, family, len(accepted), config["domain_m"])
                if scene["gap_stratum"] != ("close", "moderate", "wide")[len(accepted) % 3]:
                    continue
                if sparse_cluster_metrics(scene, domain=config["domain_m"])["minimum_gap_m"] < 0.012:
                    continue
                feasible = _preflight(scene, config, minimum_pec_cfl_fraction=0.08)
            except ValueError:
                continue
            if feasible:
                accepted.append(scene)
        validation_count = count_per_family // 8
        test_count = count_per_family // 8
        for scene in accepted:
            scene["split"] = "train"
        for split, target_gap, count in (
            ("validation", held_out_gaps[family_index][0], validation_count),
            ("test", held_out_gaps[family_index][1], test_count),
        ):
            eligible = [
                index for index, scene in enumerate(accepted)
                if scene["split"] == "train" and scene["gap_stratum"] == target_gap
            ]
            chosen = [int(rng.choice(eligible))]
            remaining = [
                index for index, scene in enumerate(accepted)
                if scene["split"] == "train" and index not in chosen
            ]
            chosen.extend(int(index) for index in rng.choice(
                remaining, size=count - 1, replace=False,
            ))
            for index in chosen:
                accepted[index]["split"] = split
        scenes.extend(accepted)
    config["purpose"] = "stage-four sparse three/four-object headroom pilot"
    config["generator"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "seed": seed,
        "count_per_family": count_per_family,
        "attempts_by_family": dict(attempts),
    }
    config["scenes"] = scenes
    config["campaign_id"] = "sparse_clusters_" + _digest({
        "generator": config["generator"], "scenes": scenes,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/sparse_cluster_pilot_32.json")
    parser.add_argument("--count-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    config = build(args.template, args.output, count_per_family=args.count_per_family, seed=args.seed)
    print(json.dumps({
        "campaign_id": config["campaign_id"],
        "scene_count": len(config["scenes"]),
        "splits": dict(Counter(scene["split"] for scene in config["scenes"])),
        "gaps": dict(Counter(scene["gap_stratum"] for scene in config["scenes"])),
        "attempts": config["generator"]["attempts_by_family"],
    }, indent=2))


if __name__ == "__main__":
    main()
