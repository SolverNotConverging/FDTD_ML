"""One geometry-first solve: python examples/solve.py [--run]."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from fdtdmesh import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable the native CUDA solve")
    parser.add_argument("--output", type=Path, default=Path("artifacts/example"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle(center=(0, 0), radius=0.14)
    sim.apply_mesh("geometry_aware", cells=(120, 120))
    sim.plot_geometry(mesh=True).savefig(args.output / "mesh.png")
    sim.save(args.output / "simulation.h5")
    print(sim.summary())
    if args.run:
        result = sim.solve(progress=True)
        result.save(args.output / "result.h5")
        result.save_plots(args.output)


if __name__ == "__main__":
    main()
