#!/usr/bin/env python3
"""Build the full exact-axis, soft-Nt campaign for the simple dielectric pool."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/simple_dielectric_factorial_pool.json")
    )
    parser.add_argument(
        "--pilot", type=Path, default=Path("configs/simple_factorial_exact_candidate_pilot.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/simple_factorial_exact_candidate_full.json")
    )
    parser.add_argument("--nt-cost-exponent", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nt_cost_exponent < 0:
        raise ValueError("nt-cost-exponent must be nonnegative")
    manifest = json.loads(args.manifest.read_text())
    pilot = json.loads(args.pilot.read_text())
    if pilot["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Pilot and manifest dataset IDs differ")
    content = {
        "schema_version": 3,
        "dataset_id": manifest["dataset_id"],
        "purpose": "full exact-axis soft-Nt labels for the simple dielectric curriculum",
        "pilot_campaign_id": pilot["campaign_id"],
        "geometry_ids": [row["geometry_id"] for row in manifest["geometries"]],
        "condition_ids": [row["task_id"] for row in manifest["conditions"]],
        "candidate_names": list(EXACT_CANDIDATES),
        "duration_schedule_s": [7e-8, 1.4e-7, 5.6e-7],
        "ranking": {
            "mode": "fixed_axis_soft_nt",
            "nt_cost_exponent": args.nt_cost_exponent,
        },
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["campaign_id"] = "simple_factorial_exact_candidate_full_" + hashlib.sha256(
        canonical
    ).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    case_count = len(content["condition_ids"]) * len(content["candidate_names"])
    print(
        f"{content['campaign_id']}: {len(content['geometry_ids'])} geometries, "
        f"{len(content['condition_ids'])} conditions, {case_count} cases -> {args.output}"
    )


if __name__ == "__main__":
    main()
