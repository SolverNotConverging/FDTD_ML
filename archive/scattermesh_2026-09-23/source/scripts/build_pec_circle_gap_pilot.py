#!/usr/bin/env python3
"""Build a preflighted circular-PEC pair pilot for exact-budget mesh policies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build_sparse_pair_campaign import _preflight
from run_sparse_pair_headroom_pilot import load_config

ROOT = Path(__file__).resolve().parents[1]


def build(template, output, *, seed=20260922):
    config = json.loads(Path(template).read_text())
    config["purpose"] = (
        "circular PEC pair gap-refinement pilot; compare exact-budget meshes, "
        "not PEC shapes"
    )
    config["policies"] = [
        {"name": "uniform", "kind": "uniform", "max_ratio": 1.0},
        {"name": "interface_wide", "kind": "interface", "interface_width_factor": 0.75,
         "interface_weight": 0.75, "max_ratio": 2.0},
        {"name": "gap_weak", "kind": "gap", "gap_width_factor": 2.0,
         "gap_min_width_m": 0.008, "gap_weight": 1.0, "max_ratio": 3.0},
        {"name": "gap_medium", "kind": "gap", "gap_width_factor": 1.0,
         "gap_min_width_m": 0.008, "gap_weight": 2.5, "max_ratio": 3.0},
        {"name": "gap_strong", "kind": "gap", "gap_width_factor": 0.5,
         "gap_min_width_m": 0.008, "gap_weight": 4.0, "max_ratio": 3.0},
    ]
    rng = np.random.default_rng(seed)
    scenes = []
    for stratum, gap in (("close", 0.022), ("moderate", 0.038), ("wide", 0.075)):
        for orientation, direction in (("horizontal", (1.0, 0.0)), ("vertical", (0.0, 1.0))):
            radius_left, radius_right = 0.06, 0.055
            half_distance = (radius_left + radius_right + gap) / 2
            for _ in range(1000):
                center = rng.uniform(0.38, 0.82, size=2)
                vector = np.asarray(direction) * half_distance
                first, second = center - vector, center + vector
                scene = {
                    "scene_id": f"pec_circles_{stratum}_{orientation}",
                    "family_id": "pec_circles",
                    "gap_stratum": stratum,
                    "incidence_angle_rad": float(rng.uniform(0, 2 * np.pi)),
                    "objects": [
                        {"shape": "circle", "center_m": first.tolist(),
                         "radius_m": radius_left, "material": {"kind": "pec"}},
                        {"shape": "circle", "center_m": second.tolist(),
                         "radius_m": radius_right, "material": {"kind": "pec"}},
                    ],
                }
                if _preflight(scene, config, minimum_pec_cfl_fraction=0.08):
                    scenes.append(scene)
                    break
            else:
                raise RuntimeError(f"No feasible {stratum} {orientation} PEC circle scene")
    config["scenes"] = scenes
    config["generator"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(Path(template).read_bytes()).hexdigest(),
        "seed": seed,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    load_config(output)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/sparse_mixed_pair_headroom_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/pec_circle_gap_pilot.json")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    config = build(args.template, args.output, seed=args.seed)
    print(f"Wrote {len(config['scenes'])} preflighted scenes to {args.output}")


if __name__ == "__main__":
    main()
