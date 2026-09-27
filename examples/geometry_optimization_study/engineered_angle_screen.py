"""Screen incidence angles for strict-conformal engineered-shape meshes.

Each row records every construction attempt. A failure is a resource-limited
construction result, not evidence that the geometry has no possible mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from fdtdmesh import Simulation
from fdtdmesh.benchmarks import ENGINEERED_SHAPES, make_engineered_geometry
from fdtdmesh.constants import C0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=ENGINEERED_SHAPES, default=ENGINEERED_SHAPES)
    parser.add_argument("--angles", nargs="+", type=float, default=(0, 15, 30, 45, 60, 90))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target-ppw", nargs="+", type=int, default=(32, 24, 40, 64))
    parser.add_argument("--max-cells", type=int, default=768)
    parser.add_argument("--max-passes", type=int, default=30)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/geometry_optimization_study/engineered_angle_feasibility.jsonl"),
    )
    args = parser.parse_args()
    if args.scale <= 0 or any(v <= 0 for v in args.target_ppw):
        parser.error("scale and target-ppw values must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name in args.shapes:
        for angle in args.angles:
            sim = Simulation(fmin=0.9e9, fmax=1.1e9)
            sim.set_geometry(make_engineered_geometry(name, scale=args.scale, incidence_deg=angle))
            row = dict(shape=name, angle=angle, scale=args.scale, attempts=[])
            for ppw in args.target_ppw:
                start = perf_counter()
                attempt = dict(target_ppw=ppw)
                try:
                    mesh = sim.apply_mesh(
                        "geometry_aware",
                        target_spacing=C0 / (1.1e9 * ppw),
                        max_cells=(args.max_cells, args.max_cells),
                        max_passes=args.max_passes,
                        time_limit=args.time_limit,
                    )
                    attempt.update(
                        status="valid",
                        cells=(mesh.Nx, mesh.Ny),
                        passes=len(mesh.metadata["geometry_aware"]["passes"]),
                        witness_anchors=[
                            len(a) for a in mesh.metadata["geometry_aware"]["witness_anchors"]
                        ],
                    )
                except (ValueError, RuntimeError) as exc:
                    attempt.update(status=type(exc).__name__, message=str(exc))
                    if hasattr(exc, "report"):
                        attempt["passes"] = len(exc.report.get("passes", []))
                attempt["seconds"] = round(perf_counter() - start, 3)
                row["attempts"].append(attempt)
                if attempt["status"] == "valid":
                    row.update(status="valid", selected_target_ppw=ppw, cells=attempt["cells"])
                    break
            else:
                row["status"] = "not_constructed"
            with args.output.open("a", encoding="utf8") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
