#!/usr/bin/env python3
"""Freeze training-selected resolution factors for held-out evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

HELD_OUT_LINEAGES = {"simple_dk_lineage_06", "simple_dk_lineage_07"}
SELECTED_CANDIDATES = (
    "region_wide_f92",
    "region_medium_f84",
    "region_medium_f87",
    "region_strong_f78",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/simple_dielectric_pool.json"))
    parser.add_argument(
        "--search", type=Path, default=Path("configs/simple_candidate_scaled_train.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/simple_candidate_scaled_holdout.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    search = json.loads(args.search.read_text())
    if search["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Search and manifest dataset IDs do not match")
    variants = {
        row["candidate"]: row for row in search.get("candidate_variants", ())
    }
    if not set(SELECTED_CANDIDATES) <= variants.keys():
        raise ValueError("Training search does not define every frozen candidate")
    geometry_ids = {
        row["geometry_id"]
        for row in manifest["geometries"]
        if row["lineage_id"] in HELD_OUT_LINEAGES
    }
    conditions = [
        row["task_id"] for row in manifest["conditions"] if row["geometry_id"] in geometry_ids
    ]
    content = {
        "schema_version": 2,
        "dataset_id": manifest["dataset_id"],
        "purpose": "held-out evaluation of resolution factors selected on training only",
        "selection_source_campaign_id": search["campaign_id"],
        "geometry_ids": sorted(geometry_ids),
        "condition_ids": conditions,
        "candidate_names": ["uniform", *SELECTED_CANDIDATES],
        "candidate_variants": [variants[name] for name in SELECTED_CANDIDATES],
        "duration_s": search["duration_s"],
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["campaign_id"] = "simple_candidate_scaled_holdout_" + hashlib.sha256(
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
