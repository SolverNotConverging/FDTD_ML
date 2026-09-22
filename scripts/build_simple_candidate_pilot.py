#!/usr/bin/env python3
"""Build the bounded first candidate-label campaign from the simple pool."""

import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST = Path("configs/simple_dielectric_pool.json")
OUTPUT = Path("configs/simple_candidate_pilot.json")
LINEAGES = {
    "simple_dk_lineage_00",
    "simple_dk_lineage_04",
    "simple_dk_lineage_06",
    "simple_dk_lineage_07",
}
CANDIDATES = (
    "uniform",
    "interface_wide",
    "region_wide",
    "region_medium",
    "region_strong",
    "hybrid_wide",
    "hybrid_medium",
    "random_density_0",
    "random_density_1",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    geometry_ids = (
        [row["geometry_id"] for row in manifest["geometries"]]
        if args.full
        else [
            row["geometry_id"]
            for row in manifest["geometries"]
            if row["lineage_id"] in LINEAGES and row["variant_index"] == 0
        ]
    )
    condition_ids = [
        row["task_id"]
        for row in manifest["conditions"]
        if row["geometry_id"] in geometry_ids and (args.full or row["cells_x"] in {32, 48, 64})
    ]
    campaign = dict(
        schema_version=1,
        dataset_id=manifest["dataset_id"],
        purpose=(
            "full simple dielectric candidate-label campaign"
            if args.full
            else "candidate-label pipeline pilot spanning train/validation/test"
        ),
        geometry_ids=geometry_ids,
        condition_ids=condition_ids,
        candidate_names=list(CANDIDATES),
        duration_s=70e-9,
        tail_gate=1e-5,
    )
    canonical = json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
    prefix = "simple_candidate_full_" if args.full else "simple_candidate_pilot_"
    campaign["campaign_id"] = prefix + hashlib.sha256(canonical).hexdigest()[:16]
    output = args.output or (Path("configs/simple_candidate_full.json") if args.full else OUTPUT)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(campaign, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(
        f"{campaign['campaign_id']}: {len(geometry_ids)} geometries, "
        f"{len(condition_ids)} conditions, {len(condition_ids) * len(CANDIDATES)} cases"
    )


if __name__ == "__main__":
    main()
