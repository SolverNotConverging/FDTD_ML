"""Resume numerical reference/mesh studies locally or on a training server."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from fdtdmesh.benchmarks import SHAPES, ReferenceSettings, plot_gallery, run_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", nargs="+", choices=SHAPES, default=["circle"])
    p.add_argument("--incidences", nargs="+", type=float, default=[0.0])
    p.add_argument("--cells", nargs=2, type=int, default=(192, 192))
    p.add_argument(
        "--strategy", choices=("differential_evolution", "powell"), default="differential_evolution"
    )
    p.add_argument("--evaluations", type=int, default=60)
    p.add_argument("--seconds", type=float, default=600.0)
    p.add_argument("--reference-ppw", nargs="+", type=int, default=(48, 72, 108, 164))
    p.add_argument("--output", type=Path, default=Path("artifacts/mesh_study"))
    a = p.parse_args()
    rows = run_sweep(
        a.shapes,
        a.incidences,
        directory=a.output,
        cells=tuple(a.cells),
        strategy=a.strategy,
        max_evaluations=a.evaluations,
        max_seconds=a.seconds,
        reference_settings=ReferenceSettings(ppw=tuple(a.reference_ppw)),
        progress=lambda row: print(row, flush=True),
    )
    plot_gallery(a.output).savefig(
        a.output / "optimized_mesh_gallery.png", dpi=130, bbox_inches="tight"
    )
    print(
        f"{sum(r['best_error'] is not None for r in rows)}/{len(rows)} cases have a feasible best mesh"
    )


if __name__ == "__main__":
    main()
