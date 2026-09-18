"""Reproduce CPML waveforms, boundary error, and fixed-collar mesh figures."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.validate_cpml import wavepacket_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, mesh, initial, final, difference, traces = wavepacket_case(45, True, pec_control=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    for lo, hi in ((0, 3), (21, 24)):
        ax.axvspan(lo, hi, color="#d9e8f4")
        ax.axhspan(lo, hi, color="#d9e8f4")
    ax.vlines(mesh.x * 1e3, 0, 24, color="#71879a", lw=0.35)
    ax.hlines(mesh.y * 1e3, 0, 24, color="#71879a", lw=0.35)
    ax.set(
        xlim=(0, 24),
        ylim=(0, 24),
        aspect="equal",
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="96 x 96 cells; fixed 12-cell collars per side",
    )
    t = traces["times"] * 1e12
    ax = axes[0, 1]
    ax.plot(t, traces["reference"][:, 2], color="#3274ad", label="Enlarged reference")
    ax.plot(t, traces["small"][:, 2], "--", color="#d65f32", label="CPML")
    ax.set(xlabel="Time (ps)", ylabel="Ez (V/m)", title="45-degree packet / diagonal receiver")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    denominator = np.max(abs(traces["reference"]))
    ax.semilogy(
        t,
        np.maximum(np.max(abs(traces["difference"]), axis=1) / denominator, 1e-15),
        color="#d65f32",
    )
    ax.axhline(0.01, color="#343a40", ls="--", label="1% acceptance threshold")
    ax.set(
        xlabel="Time (ps)",
        ylabel="Peak error across 9 receivers / reference peak",
        title=f"Measured peak: {report['peak_waveform_error']:.2e}",
        ylim=(1e-12, 1),
    )
    ax.legend(frameon=False)
    ax = axes[1, 1]
    a = 12
    image = ax.pcolormesh(
        mesh.x[a:-a] * 1e3,
        mesh.y[a:-a] * 1e3,
        difference[a:-a, a:-a].T,
        cmap="RdBu_r",
        shading="auto",
    )
    ax.set(
        aspect="equal", xlabel="x (mm)", ylabel="y (mm)", title="Final interior Ez difference (V/m)"
    )
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle("CFS-CPML on a graded nonuniform mesh", fontsize=17)
    fig.text(
        0.5,
        0.015,
        f"Vacuum, 60 GHz carrier; errors against the same interior grid extended outward. "
        f"PEC control peak error: {report['pec_control_peak_error']:.1%}.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"cpml_validation.{suffix}", dpi=160)
    plt.close(fig)
    print(report)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
