"""Run a GPU-resident PEC scattering example and save its results."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import fdtdmesh
from fdtdmesh.api import DFTConvergence, Simulation, SolverSettings


def _add_shape(simulation, shape, lam):
    x = y = 3 * lam
    if shape == "cylinder":
        simulation.add_circle((x, y), 0.47 * lam, material="PEC")
    elif shape == "rectangle":
        simulation.add_rectangle(
            (x - 0.43 * lam, x + 0.43 * lam),
            (y - 0.37 * lam, y + 0.37 * lam),
            material="PEC",
        )
    elif shape == "pair":
        simulation.add_circle((x - 0.4 * lam, y), 0.27 * lam, material="PEC")
        simulation.add_circle((x + 0.4 * lam, y), 0.27 * lam, material="PEC")
    elif shape == "slot":
        simulation.add_rectangle(
            (x - 0.5 * lam, x + 0.5 * lam),
            (y - 0.45 * lam, y + 0.45 * lam),
            material="PEC",
        )
        simulation.add_rectangle(
            (x - 0.13 * lam, x + 0.13 * lam),
            (y - 0.15 * lam, y + 0.5 * lam),
            material="air",
        )
    elif shape != "empty":
        raise ValueError(f"Unknown canonical geometry {shape!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", choices=["cylinder", "rectangle", "pair", "slot", "empty"], default="cylinder"
    )
    parser.add_argument("--ppw", type=int, default=24, help="Nominal cells per wavelength")
    parser.add_argument("--graded", action="store_true", help="Alias for --strategy deterministic")
    parser.add_argument(
        "--strategy", choices=["uniform", "deterministic", "cnn"], default="uniform"
    )
    parser.add_argument("--checkpoint", type=Path, help="CNN checkpoint for --strategy cnn")
    parser.add_argument("--frequency", type=float, default=1e9, help="Pulse centre in Hz")
    parser.add_argument(
        "--bins", type=float, nargs="+", help="DFT frequencies in Hz; default is a 21-bin band"
    )
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--check-interval", type=int, default=2048)
    parser.add_argument(
        "--max-steps", type=int, default=200000, help="Failure safety cap, not duration"
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/example"))
    args = parser.parse_args()

    if not np.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("--frequency must be finite and positive")
    if args.ppw < 1 or args.ppw != int(args.ppw):
        parser.error("--ppw must be a positive integer")
    if args.bins is not None:
        bins = tuple(args.bins)
        if (
            not bins
            or not np.isfinite(bins).all()
            or min(bins) <= 0
            or any(a >= b for a, b in zip(bins, bins[1:]))
        ):
            parser.error("--bins must be increasing, finite, positive frequencies")
        dft_bins = bins
    else:
        dft_bins = 21

    f0 = args.frequency
    half_band = max(0.1 * f0, max(abs(f - f0) for f in args.bins) if args.bins else 0)
    fmin, fmax = f0 - half_band, f0 + half_band
    if not 0 < fmin < fmax:
        parser.error("Bins are too far from --frequency for a positive symmetric analysis band")

    lam = fdtdmesh.C0 / f0
    settings = SolverSettings(
        dft_bins=dft_bins,
        stop=DFTConvergence(
            check_interval=args.check_interval, max_steps=args.max_steps, rtol=args.rtol
        ),
        precision=args.precision,
    )
    simulation = Simulation(size=(6 * lam, 6 * lam), fmin=fmin, fmax=fmax, settings=settings)
    _add_shape(simulation, args.shape, lam)

    strategy = "deterministic" if args.graded else args.strategy
    if args.graded and args.strategy == "cnn":
        parser.error("--graded cannot be combined with --strategy cnn")
    if strategy == "cnn" and args.checkpoint is None:
        parser.error("--checkpoint is required with --strategy cnn")
    if strategy != "cnn" and args.checkpoint is not None:
        parser.error("--checkpoint is only accepted with --strategy cnn")
    print("Preparing geometry and constrained mesh...", flush=True)
    simulation.apply_mesh(
        strategy,
        cells=(6 * args.ppw, 6 * args.ppw),
        checkpoint=args.checkpoint if strategy == "cnn" else None,
    )
    result = simulation.solve(progress=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(args.output / "scattering.h5")
    result.save_plots(args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
