"""Plot CNN density quantiles and the closest mesh with mandatory 1.4 grading."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.mesh import axis_mesh
from fdtdmesh.mesh_projection import density_quantiles


def ratios(x):
    h = np.diff(x)
    return np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/anchor_grading"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("Close anchors / uniform preference", 40, [1], [0.0098, 0.0102]),
        ("Close anchors / adaptive preference", 40, [1, 1, 6, 6, 1, 1], [0.0098, 0.0102]),
        ("Recovered allocation counterexample", 10, [1, 100], [0.01]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 9), gridspec_kw={"height_ratios": [1, 1.5, 1.5]})
    records = []
    for col, (title, n, rho, anchors) in enumerate(cases):
        target = density_quantiles(0.02, n, rho)
        mesh, stats = axis_mesh(0.02, n, rho, anchors, return_diagnostics=True)
        assert all(a in mesh for a in anchors) and len(mesh) == n + 1
        assert ratios(mesh).max() <= 1.4 * (1 + 1e-8)
        stats.update(
            case=title,
            cells=n,
            anchors_m=anchors,
            target_m=target.tolist(),
            projected_m=mesh.tolist(),
            maximum_ratio=float(ratios(mesh).max()),
            interval_cells=np.diff(
                [0] + [int(np.flatnonzero(mesh == a)[0]) for a in anchors] + [n]
            ).tolist(),
        )
        records.append(stats)
        a = axes[0, col]
        a.set_title(title + f"\n{n} cells", fontsize=12)
        for x, y in zip(target, mesh):
            a.plot([x * 1e3, y * 1e3], [1, 0], color="#cbd2d9", lw=0.7)
        a.vlines(target * 1e3, 0.8, 1.2, color="#3274ad", lw=1)
        a.vlines(mesh * 1e3, -0.2, 0.2, color="#d65f32", lw=1)
        a.set_yticks([0, 1], ["Legal mesh", "CNN target"])
        a.set_ylim(-0.4, 1.4)
        a = axes[1, col]
        a.stairs(np.diff(target) * 1e3, target * 1e3, color="#3274ad", label="Density quantiles")
        a.stairs(np.diff(mesh) * 1e3, mesh * 1e3, color="#d65f32", label="Constrained optimum")
        a.set_ylabel("Cell width (mm)")
        a.set_ylim(bottom=0)
        a.legend(fontsize=9, frameon=False)
        a = axes[2, col]
        a.plot(target[1:-1] * 1e3, ratios(target), ".-", color="#3274ad")
        a.plot(mesh[1:-1] * 1e3, ratios(mesh), ".-", color="#d65f32")
        a.axhline(1.4, color="#343a40", ls="--", label="Hard limit 1.4")
        a.set_yscale("log")
        a.set_ylim(0.9, max(1.8, ratios(target).max() * 1.2))
        a.set_ylabel("Adjacent width ratio (log)")
        a.set_xlabel("x (mm)")
        a.legend(fontsize=9, frameon=False)
        for a in axes[:, col]:
            for anchor in anchors:
                a.axvline(anchor * 1e3, color="#8a3f88", ls=":", lw=1)
            a.set_xlim(0, 20)
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Mandatory grading moves lines and optimizes anchor interval cell counts", fontsize=17
    )
    fig.text(
        0.5,
        0.012,
        "Purple = exact anchors. Blue = unconstrained density preference (may miss anchors). "
        "Orange = minimum normalized line L1 displacement under all hard constraints.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"anchor_grading_comparison.{suffix}", dpi=160)
    plt.close(fig)
    (args.output_dir / "measurements.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    for record in records:
        print(
            record["case"],
            "ratio:",
            record["maximum_ratio"],
            "allocation:",
            record["interval_cells"],
        )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
