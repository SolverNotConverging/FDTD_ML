#!/usr/bin/env python3
"""Build the training-only search for matched-update candidate resolutions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

LINEAGES = {
    "simple_dk_lineage_03",
    "simple_dk_lineage_04",
    "simple_dk_lineage_05",
}
VARIANT_INDICES = {0, 2}
SCALED_CANDIDATES = {
    "region_wide": (0.92, 0.95, 0.98),
    "region_medium": (0.84, 0.87, 0.90),
    "region_strong": (0.78, 0.81, 0.84),
    "hybrid_wide": (0.90, 0.93, 0.96),
    "hybrid_medium": (0.80, 0.84, 0.88),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/simple_dielectric_pool.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("configs/simple_candidate_scaled_train.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    geometry_ids = {
        row["geometry_id"]
        for row in manifest["geometries"]
        if row["lineage_id"] in LINEAGES and row["variant_index"] in VARIANT_INDICES
    }
    conditions = [
        row["task_id"] for row in manifest["conditions"] if row["geometry_id"] in geometry_ids
    ]
    variants = []
    for base_candidate, factors in SCALED_CANDIDATES.items():
        for factor in factors:
            variants.append(
                {
                    "candidate": f"{base_candidate}_f{round(100 * factor):02d}",
                    "base_candidate": base_candidate,
                    "cell_factor": factor,
                }
            )
    content = {
        "schema_version": 2,
        "dataset_id": manifest["dataset_id"],
        "purpose": "training-only search for nonuniform candidate resolution factors",
        "geometry_ids": sorted(geometry_ids),
        "condition_ids": conditions,
        "candidate_names": ["uniform", *[row["candidate"] for row in variants]],
        "candidate_variants": variants,
        "duration_s": 7e-8,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["campaign_id"] = "simple_candidate_scaled_train_" + hashlib.sha256(canonical).hexdigest()[
        :16
    ]
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
