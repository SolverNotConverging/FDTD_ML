#!/usr/bin/env python3
"""Materialize the frozen analytic-circle position/scale evaluation plan."""

import argparse
import json
import os
from pathlib import Path

from scattermesh.generalization import circle_generalization_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/circle_position_scale_generalization.json"),
    )
    args = parser.parse_args()
    payload = circle_generalization_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "evaluation_id": payload["evaluation_id"],
                "example_count": len(payload["examples"]),
                "solver_case_count": 2 * len(payload["examples"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
