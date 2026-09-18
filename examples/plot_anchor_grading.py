"""Audit current anchor grading without changing mesher behavior.

Run: python examples/plot_anchor_grading.py
Outputs PNG/SVG plots and measured JSON in artifacts/anchor_grading/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.mesh import AxisConstraints, MeshInfeasibleError, axis_mesh

RAW = "#3274ad"
GRADED = "#d65f32"
ANCHOR = "#8a3f88"


def ratios(lines):
    h = np.diff(lines)
    return np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e7eb", linewidth=0.7)
    ax.set_axisbelow(True)


def comparison(output):
    length, count, limit = 0.02, 40, 1.3
    anchors = [0.0098, 0.0102]
    cases = [("Uniform density", [1]), ("Adaptive density (1, 1, 6, 6, 1, 1)", [1, 1, 6, 6, 1, 1])]
    fig, axes = plt.subplots(
        3, 2, figsize=(13.2, 9.5), sharex="col", gridspec_kw={"height_ratios": [1, 1.55, 1.25]}
    )
    records = []
    for col, (title, density) in enumerate(cases):
        raw = axis_mesh(length, count, density, anchors)
        graded = axis_mesh(length, count, density, anchors, AxisConstraints(max_ratio=limit))
        hraw, hgraded = np.diff(raw), np.diff(graded)
        moved = int(np.count_nonzero(np.abs(graded - raw) > 1e-6))
        assert len(raw) == len(graded) == count + 1
        assert all(a in graded for a in anchors)
        assert ratios(graded).max() <= limit * (1 + 1e-8)
        # Confirm lines on BOTH sides of the close anchors actually move.
        delta = np.abs(graded - raw)
        assert np.any(delta[raw < anchors[0]] > 1e-10)
        assert np.any(delta[raw > anchors[-1]] > 1e-10)
        record = {
            "case": title,
            "cells": count,
            "anchors_m": anchors,
            "max_ratio_requested": limit,
            "ungraded_max_ratio": float(ratios(raw).max()),
            "graded_max_ratio": float(ratios(graded).max()),
            "lines_moved_over_1_um": moved,
            "max_displacement_m": float(np.max(np.abs(graded - raw))),
            "ungraded_min_spacing_m": float(hraw.min()),
            "graded_min_spacing_m": float(hgraded.min()),
            "ungraded_lines_m": raw.tolist(),
            "graded_lines_m": graded.tolist(),
        }
        records.append(record)
        ax = axes[0, col]
        ax.set_title(f"{title}\n{count} cells · anchors at 9.8 and 10.2 mm", fontsize=12, pad=14)
        for x, y in zip(raw, graded):
            ax.plot([x * 1e3, y * 1e3], [1, 0], color="#cbd2d9", lw=0.65, zorder=1)
        ax.vlines(raw * 1e3, 0.83, 1.17, color=RAW, lw=1.2, zorder=3)
        ax.vlines(graded * 1e3, -0.17, 0.17, color=GRADED, lw=1.2, zorder=3)
        ax.set_yticks([0, 1], ["Graded", "Default"])
        ax.set_ylim(-0.4, 1.5)
        ax.text(
            0.02,
            0.95,
            f"{moved} of {count + 1} lines moved > 1 μm",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            color="#3e4c59",
        )
        ax = axes[1, col]
        ax.stairs(hraw * 1e3, raw * 1e3, color=RAW, lw=1.8, label="Current default: no grading")
        ax.stairs(
            hgraded * 1e3, graded * 1e3, color=GRADED, lw=2, label="Explicit max_ratio = 1.30"
        )
        ax.set_ylabel("Cell width Δx (mm)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8.8, frameon=False)
        ax = axes[2, col]
        ax.plot(raw[1:-1] * 1e3, ratios(raw), ".-", color=RAW, lw=1.4, markersize=4)
        ax.plot(graded[1:-1] * 1e3, ratios(graded), ".-", color=GRADED, lw=1.4, markersize=4)
        ax.axhline(limit, color="#333c44", ls="--", lw=1)
        ax.set_ylabel("Adjacent width ratio\nmax(hᵢ₊₁/hᵢ, hᵢ/hᵢ₊₁)")
        ax.set_xlabel("x (mm)")
        ax.set_ylim(0.9, max(1.6, ratios(raw).max() * 1.18))
        ax.text(
            0.03,
            0.93,
            f"Maximum: {ratios(raw).max():.2f} → {ratios(graded).max():.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
        )
        for ax in axes[:, col]:
            for anchor in anchors:
                ax.axvline(anchor * 1e3, color=ANCHOR, lw=1, ls=":", alpha=0.85, zorder=2)
            ax.set_xlim(0, 20)
            style_axis(ax)
    fig.suptitle(
        "Anchors can trigger abrupt spacing; explicit grading moves neighboring lines",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Purple dotted lines = fixed anchors. Gray links = movement of the same mesh line.\n"
        "Cell counts and anchor locations are identical before and after projection. Grading is currently opt-in.",
        ha="center",
        fontsize=10,
        color="#43515d",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955), h_pad=1.6)
    for ext in ("png", "svg"):
        fig.savefig(output / f"anchor_grading_comparison.{ext}", dpi=170, facecolor="white")
    plt.close(fig)
    return records


def allocation_limit(output):
    raw = axis_mesh(0.02, 10, [1, 100], [0.01])
    failure = None
    try:
        axis_mesh(0.02, 10, [1, 100], [0.01], AxisConstraints(max_ratio=1.3))
    except MeshInfeasibleError as error:
        failure = str(error)
    assert failure is not None
    # Independent feasible witness for the SAME hard constraints, not a second
    # successful run with the original density. Its soft density objective differs.
    alternative = np.linspace(0, 0.02, 11)
    assert 0.01 in alternative and ratios(alternative).max() <= 1.3
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]}
    )
    axes[0].vlines(raw * 1e3, 0.85, 1.15, color=RAW, lw=2)
    axes[0].vlines(alternative * 1e3, -0.15, 0.15, color=GRADED, lw=2)
    axes[0].set_yticks(
        [0, 1], ["Feasible alternative: 5 + 5 cells", "Current allocation: 1 + 9 cells"]
    )
    axes[0].set_ylim(-0.35, 1.4)
    axes[1].stairs(
        np.diff(raw) * 1e3,
        raw * 1e3,
        color=RAW,
        lw=2,
        label="Density target [1, 100]: projection fails at r = 1.30",
    )
    axes[1].stairs(
        np.diff(alternative) * 1e3,
        alternative * 1e3,
        color=GRADED,
        lw=2,
        label="Alternative allocation: all widths 2 mm; ratio = 1.00",
    )
    axes[1].set_ylabel("Cell width Δx (mm)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylim(0, 12)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        ax.axvline(10, color=ANCHOR, ls=":", lw=1.5)
        ax.set_xlim(0, 20)
        style_axis(ax)
    fig.suptitle(
        "Current limitation: smoothing does not revisit interval cell counts",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Both layouts have 10 cells and the anchor at 10 mm. The orange layout is a feasibility witness,\n"
        "not the current mesher's output for density [1, 100]. A grading-aware allocation could recover it.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(output / f"anchor_allocation_limit.{ext}", dpi=170, facecolor="white")
    plt.close(fig)
    return {
        "density": [1, 100],
        "current_interval_counts": [1, 9],
        "feasible_alternative_interval_counts": [5, 5],
        "error": failure,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/anchor_grading"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    summary = {
        "comparisons": comparison(args.output_dir),
        "allocation_limitation": allocation_limit(args.output_dir),
    }
    (args.output_dir / "measurements.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for record in summary["comparisons"]:
        print(
            f"{record['case']}: ratio {record['ungraded_max_ratio']:.6f} -> "
            f"{record['graded_max_ratio']:.6f}, {record['lines_moved_over_1_um']} lines moved > 1 μm"
        )
    print(f"Plots and measurements: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
