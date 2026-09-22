"""Small repeatable CUDA timing experiment; reports work at fixed physical duration."""

import argparse
import json
from statistics import median

import numpy as np

from fdtdmesh import FDTD_2D_Ez


def benchmark(nx, ny, duration, repeats, nonuniform):
    sim = FDTD_2D_Ez(0.02, 0.015, nx, ny, 20e9, t_end=duration)
    sim.add_source("point", x=0.003, y=0.007, width=2e-11, delay=8e-11)
    sim.add_receiver("point", x=0.016, y=0.007)
    if nonuniform:
        u = (np.arange(128) + 0.5) / 128
        sim.mesh_from_density(1 + 3 * np.exp(-(((u - 0.5) / 0.2) ** 2)), np.ones(128))
    else:
        sim.mesh_uniform()
    sim.run()  # Warm up context, kernels and driver.
    runs = [sim.run().diagnostics for _ in range(repeats)]
    report = runs[-1].copy()
    report["gpu_ms_median"] = median(run["gpu_ms"] for run in runs)
    report["wall_seconds_median"] = median(run["wall_seconds"] for run in runs)
    report["cell_updates_per_second"] = report["cell_updates"] / (report["gpu_ms_median"] * 1e-3)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--duration", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    print(
        json.dumps(
            {
                name: benchmark(args.nx, args.ny, args.duration, args.repeats, adaptive)
                for name, adaptive in (("uniform", False), ("nonuniform", True))
            },
            indent=2,
        )
    )
