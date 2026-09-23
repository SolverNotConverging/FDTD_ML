#!/usr/bin/env python3
"""Write exact-budget candidate search for the second circle remediation stage."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/circle_corner_remediation_pool.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/circle_corner_remediation_candidate.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    content = {
        "schema_version": 3,
        "dataset_id": manifest["dataset_id"],
        "purpose": "exact-budget large lossy corner-circle remediation labels",
        "geometry_ids": [row["geometry_id"] for row in manifest["geometries"]],
        "condition_ids": [row["task_id"] for row in manifest["conditions"]],
        "candidate_names": list(EXACT_CANDIDATES),
        "duration_schedule_s": [70e-9, 140e-9, 560e-9, 1.12e-6, 2.24e-6],
        "pml_thickness_m": 0.12,
        "monitor_policy": "widest_non_pml_enclosing",
        "ranking": {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1},
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["campaign_id"] = "circle_corner_remediation_candidate_" + hashlib.sha256(
        canonical
    ).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "campaign_id": content["campaign_id"],
                "condition_count": len(content["condition_ids"]),
                "case_count": len(content["condition_ids"]) * len(EXACT_CANDIDATES),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
