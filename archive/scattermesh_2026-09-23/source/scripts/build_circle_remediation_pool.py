#!/usr/bin/env python3
"""Write the disjoint translated/scaled circle-remediation manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from scattermesh import circle_remediation_pool, simple_candidate_tasks
from scattermesh.curriculum import REMEDIATION_BUDGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/circle_remediation_pool.json")
    )
    args = parser.parse_args()
    geometries = circle_remediation_pool()
    conditions = simple_candidate_tasks(geometries, budgets=REMEDIATION_BUDGETS)
    content = {
        "schema_version": 1,
        "purpose": "disjoint translated/scaled circle remediation after frozen OOD failure",
        "geometry_count": len(geometries),
        "condition_count": len(conditions),
        "geometries": geometries,
        "conditions": conditions,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["dataset_id"] = "circle_remediation_" + hashlib.sha256(canonical).hexdigest()[:16]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "dataset_id": content["dataset_id"],
                "geometry_count": len(geometries),
                "condition_count": len(conditions),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
