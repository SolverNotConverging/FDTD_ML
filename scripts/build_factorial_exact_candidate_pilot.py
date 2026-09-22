#!/usr/bin/env python3
"""Build the exact-axis, soft-Nt pilot on the decorrelated scene pool."""

import argparse
import hashlib
import json
import os
from pathlib import Path

EXACT_CANDIDATES = (
    "uniform",
    "interface_wide",
    "region_wide",
    "region_medium",
    "region_strong",
    "hybrid_wide",
)
PILOT_GEOMETRIES = (
    ("train", 2.0, 0.050, "sigma_e_s_per_m", 0.00, 0),
    ("train", 8.0, 0.120, "sigma_e_s_per_m", 0.06, 1),
    ("train", 30.0, 0.120, "loss_tangent_at_1ghz", 0.10, 1),
    ("validation", 26.0, 0.1025, "loss_tangent_at_1ghz", 0.14, 0),
    ("test", 28.0, 0.110, "loss_tangent_at_1ghz", 0.18, 0),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/simple_dielectric_factorial_pool.json")
    )
    parser.add_argument(
        "--evidence", type=Path, default=Path("configs/simple_candidate_full.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/simple_factorial_exact_candidate_pilot.json")
    )
    parser.add_argument("--nt-cost-exponent", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nt_cost_exponent < 0:
        raise ValueError("nt-cost-exponent must be nonnegative")
    manifest = json.loads(args.manifest.read_text())
    evidence = json.loads(args.evidence.read_text())
    selected = []
    for (
        split,
        epsilon_r,
        base_radius,
        material_field,
        material_value,
        variant_index,
    ) in PILOT_GEOMETRIES:
        scale = 0.97 if variant_index == 0 else 1.03
        matches = [
            row
            for row in manifest["geometries"]
            if row["split"] == split
            and row["epsilon_r"] == epsilon_r
            and abs(row["radius_m"] - base_radius * scale) < 1e-12
            and abs(row[material_field] - material_value) < 1e-12
            and row["variant_index"] == variant_index
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one pilot geometry, found {len(matches)}")
        selected.append(matches[0]["geometry_id"])
    geometry_ids = set(selected)
    conditions = [
        row["task_id"] for row in manifest["conditions"] if row["geometry_id"] in geometry_ids
    ]
    content = {
        "schema_version": 3,
        "dataset_id": manifest["dataset_id"],
        "purpose": "exact-axis soft-Nt pilot on decorrelated simple dielectric pool",
        "same_axis_evidence_campaign_id": evidence["campaign_id"],
        "geometry_ids": selected,
        "condition_ids": conditions,
        "candidate_names": list(EXACT_CANDIDATES),
        "duration_schedule_s": [7e-8, 1.4e-7, 5.6e-7],
        "ranking": {
            "mode": "fixed_axis_soft_nt",
            "nt_cost_exponent": args.nt_cost_exponent,
        },
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["campaign_id"] = "simple_factorial_exact_candidate_pilot_" + hashlib.sha256(
        canonical
    ).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(
        f"{content['campaign_id']}: {len(geometry_ids)} geometries, "
        f"{len(conditions)} conditions, "
        f"{len(conditions) * len(content['candidate_names'])} cases -> {args.output}"
    )


if __name__ == "__main__":
    main()
