#!/usr/bin/env python3
"""Write the decorrelated simple-dielectric pool and condition manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from scattermesh import factorial_simple_dielectric_pool, simple_candidate_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("configs/simple_dielectric_factorial_pool.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    geometries = factorial_simple_dielectric_pool()
    conditions = simple_candidate_tasks(geometries)
    content = {
        "schema_version": 1,
        "purpose": "decorrelated simple dielectric pool before CNN training",
        "geometry_count": len(geometries),
        "condition_count": len(conditions),
        "geometries": geometries,
        "conditions": conditions,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["dataset_id"] = "simple_factorial_" + hashlib.sha256(canonical).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(
        f"{content['dataset_id']}: {content['geometry_count']} geometries, "
        f"{content['condition_count']} conditions -> {args.output}"
    )


if __name__ == "__main__":
    main()
