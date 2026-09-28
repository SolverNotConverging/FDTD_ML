"""Configured preparation/reference/search; GPU work requires --run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fdtdmesh import BoundaryPolicy, Simulation
from fdtdmesh.catalog import make_geometry
from fdtdmesh.optimization import (
    ReferenceSettings,
    SearchSettings,
    optimize_mesh,
    qualify_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in config["shapes"]:
        for angle in config["orientations_deg"]:
            sim = Simulation(
                fmin=config["fmin"],
                fmax=config["fmax"],
                boundary=BoundaryPolicy(**config.get("boundary", {})),
            )
            sim.set_geometry(make_geometry(shape, scale=config["scale"], orientation_deg=angle))
            case_dir = output / f"{shape}_{angle:g}"
            case_dir.mkdir(parents=True, exist_ok=True)
            reference = None
            for cells in config["budgets"]:
                name = f"{cells[0]}x{cells[1]}"
                row = dict(shape=shape, orientation_deg=angle, cells=cells, status="preparing")
                try:
                    seed = sim.apply_mesh("geometry_aware", cells=tuple(cells))
                    sim.plot_geometry(mesh=True).savefig(case_dir / f"{name}_seed.png")
                    plt.close("all")
                    row.update(status="prepared", boundary=sim.discretization.boundary_report)
                    if args.run:
                        if reference is None:
                            reference = qualify_reference(
                                sim,
                                directory=case_dir / "reference",
                                settings=ReferenceSettings(**config.get("reference", {})),
                                initial_mesh=seed,
                            )
                        row["reference_status"] = reference.report["status"]
                        if reference.qualified:
                            optimized = optimize_mesh(
                                sim,
                                reference,
                                cells=tuple(cells),
                                directory=case_dir / name,
                                initial_mesh=seed,
                                settings=SearchSettings(**config.get("search", {})),
                            )
                            row.update(
                                status=optimized.report["status"],
                                error=optimized.report["best_error"],
                                statistics=optimized.report.get("search_statistics"),
                                archive=str(optimized.directory),
                            )
                            if optimized.best is not None:
                                plot = case_dir / f"{name}_best.png"
                                optimized.best.plot_geometry(mesh=True).savefig(plot)
                                row["figure"] = str(plot.relative_to(output))
                                plt.close("all")
                except (ValueError, RuntimeError) as exc:
                    row.update(status="failed", error_message=str(exc))
                rows.append(row)
                (output / "summary.json").write_text(
                    json.dumps(dict(config=config, cases=rows), indent=2)
                )
                print(shape, angle, name, row["status"], flush=True)


if __name__ == "__main__":
    main()
