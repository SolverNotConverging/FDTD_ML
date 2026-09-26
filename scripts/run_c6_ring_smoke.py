#!/usr/bin/env python3
"""Run and preserve a settled schema-4 dielectric-ring TMz smoke case."""

import argparse
import json
from pathlib import Path

from scattermesh.simulation_v2 import evaluate_case
from scattermesh.topology_v2 import square_ring_smoke_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs_v2/c6_ring_smoke_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", type=int, default=64)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("New FDTD campaign runs require the compiled CUDA kernel")

    output = Path(args.output)
    scene = square_ring_smoke_scene()
    record, cached = evaluate_case(
        scene,
        args.cells,
        0.0,
        "uniform",
        output / "uniform",
        device=args.device,
        pml_thickness=0.12,
        material_samples=12,
        tail_tolerance=1e-5,
    )
    report = {
        "schema_version": 1,
        "campaign": "c6_schema4_dielectric_ring_smoke_v1",
        "device": args.device,
        "scene": scene,
        "cells_per_axis": args.cells,
        "policy": "uniform",
        "incidence_angle_rad": 0.0,
        "reference_comparison": None,
        "interpretation": "Geometry and solver compatibility only; no reference accuracy or mesh-saving claim.",
        "cached_case": cached,
        "result": record,
    }
    report_path = output / "c6_ring_smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        {
            "status": record["status"],
            "accepted": record["accepted"],
            "tail_ratio": record.get("tail_peak_over_global_peak"),
            "dt": record.get("dt"),
            "Nt": record.get("Nt"),
            "wall_seconds": record.get("wall_seconds"),
            "report": str(report_path),
        }
    )


if __name__ == "__main__":
    main()
