"""Run a GPU-resident PEC scattering example and save its far-field results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from fdtdmesh.cases import canonical_case
from fdtdmesh.constants import C0
from fdtdmesh.simulation import Convergence, ConvergenceError, run_scattering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", choices=["cylinder", "rectangle", "pair", "slot", "empty"], default="cylinder"
    )
    parser.add_argument("--ppw", type=int, default=24, help="Nominal budget; multiple of eight")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--frequency", type=float, default=1e9, help="Pulse centre in Hz")
    parser.add_argument(
        "--bins", type=float, nargs="+", help="DFT frequencies in Hz; default pulse centre"
    )
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--check-interval", type=int, default=2048)
    parser.add_argument(
        "--max-steps", type=int, default=200000, help="Failure safety cap, not duration"
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/example"))
    args = parser.parse_args()
    policy = Convergence(
        max_steps=args.max_steps, check_interval=args.check_interval, rtol=args.rtol
    )
    print("Preparing geometry and constrained mesh...", flush=True)
    case, mesh = canonical_case(
        args.shape,
        ppw=args.ppw,
        nonuniform=args.graded,
        frequency=args.frequency,
        frequencies=tuple(args.bins or ()),
        convergence=policy,
    )

    def progress(report):
        print(
            f"step {report['step']}: DFT error/tolerance={report['dft_error_ratio']:.3g}, "
            f"residual={report['residual']:.3g}, stable={report['stable_checks']}",
            flush=True,
        )

    try:
        result = run_scattering(case, mesh, dtype=args.precision, progress=progress)
    except ConvergenceError as exc:
        exc.result.save(args.output / "unconverged.npz")
        raise SystemExit(str(exc)) from exc
    result.save(args.output / "scattering.npz")
    (args.output / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")
    lam = C0 / case.frequency
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    for x in mesh.x / lam:
        ax.axvline(x, color=".85", lw=0.4)
    for y in mesh.y / lam:
        ax.axhline(y, color=".85", lw=0.4)
    for kind, metal, data in case.scene.primitives:
        color = "#25384a" if metal else "white"
        if kind == "circle":
            patch = Circle(np.asarray(data[:2]) / lam, data[2] / lam, color=color)
        else:
            patch = Polygon(np.asarray(data) / lam, color=color)
        ax.add_patch(patch)
    ax.set(
        xlim=(2, 4),
        ylim=(2, 4),
        aspect="equal",
        xlabel="x / λ",
        ylabel="y / λ",
        title=f"Continuous PEC geometry; {result.diagnostics['enlarged_pairs']} enlarged pairs",
    )
    for f, frequency in enumerate(result.frequencies):
        axes[0, 1].plot(
            np.rad2deg(result.angles),
            result.width[f] / (C0 / frequency),
            label=f"{frequency / 1e9:g} GHz",
        )
    axes[0, 1].set(
        xlabel="Observation angle (degrees)", ylabel="Scattering width / λ", title="GPU NF2FF"
    )
    axes[0, 1].legend()
    axes[1, 0].semilogy(
        result.history[:, 0],
        np.maximum(result.history[:, 2], 1e-16),
        "o-",
        label="DFT error / tolerance",
    )
    axes[1, 0].semilogy(
        result.history[:, 0],
        result.history[:, 3] / policy.field_tol,
        "o-",
        label="Residual / tolerance",
    )
    axes[1, 0].axhline(1, color="black", ls="--", lw=0.7)
    axes[1, 0].set(
        xlabel="Update count", ylabel="Convergence ratio", title="Device stopping checks"
    )
    axes[1, 0].legend()
    for name in ("x", "y"):
        lines = getattr(mesh, name)
        axes[1, 1].plot((lines[:-1] + lines[1:]) / (2 * lam), np.diff(lines) / lam, label=name)
    axes[1, 1].set(xlabel="Position / λ", ylabel="Cell width / λ", title="Axis spacing")
    axes[1, 1].set_ylim(0, 1.1 * max(np.diff(mesh.x).max(), np.diff(mesh.y).max()) / lam)
    axes[1, 1].legend()
    fig.savefig(args.output / "scattering.png", dpi=160)
    print(
        f"Converged after {result.diagnostics['Nt']} updates; GPU {result.diagnostics['gpu_ms']:.1f} ms."
    )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
