"""Render reproducible sparse geometry examples, without reference simulations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.data.generate_v7 import feature_audit, make_sparse_scene
from fdtdmesh.data.schema import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sparse_v7_preview"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = [
        make_sparse_scene(2026, "validation", family + 4 * size + 12 * ((family + size) % 4))
        for family in range(4)
        for size in range(3)
    ]
    write_manifest(
        args.output / "manifest.json",
        scenes,
        generation=dict(version=7, purpose="geometry_preview"),
    )
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    for ax, spec in zip(axes.flat, scenes):
        sim = spec.build([128, 128])
        coordinates = [(np.arange(512) + 0.5) * d / 512 for d in spec.domain]
        eps, _, _, pec = sim.sample(*coordinates)
        im = ax.imshow(
            np.ma.masked_where(pec.T, eps.T),
            origin="lower",
            interpolation="nearest",
            extent=[0, sim.Lx * 1000, 0, sim.Ly * 1000],
            cmap=cmap,
            vmin=1,
            vmax=30,
        )
        for probe in spec.sources:
            ax.plot(probe["x"] * 1000, probe["y"] * 1000, "*", color="red", mec="white", ms=8)
        for probe in spec.receivers:
            ax.plot(probe["x"] * 1000, probe["y"] * 1000, "x", color="cyan", ms=5)
        x0, x1, y0, y1 = sim.pml.interfaces(sim)
        ax.plot(
            np.array([x0, x1, x1, x0, x0]) * 1000,
            np.array([y0, y0, y1, y1, y0]) * 1000,
            "--",
            color="#ffd166",
            lw=0.7,
        )
        audit = feature_audit(spec)
        detail = f"occupied area {audit['occupancy_fraction']:.1%}"
        if audit["gap_pixels"] is not None:
            detail += f" · gap {audit['gap_pixels']:.0f} input pixels"
        ax.set_title(spec.family.replace("_", " ") + "\n" + detail, fontsize=9)
        ax.set(xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=list(axes.flat), shrink=0.65, label="Relative permittivity εr · black: PEC")
    fig.suptitle(
        "Sparse supplement: small / medium / large objects\n"
        "Red star: source · cyan ×: receivers · yellow dashed: PML interface",
        fontsize=14,
    )
    fig.savefig(args.output / "geometry_gallery.png", dpi=150)
    print(args.output / "geometry_gallery.png")


if __name__ == "__main__":
    main()
