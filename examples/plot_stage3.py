"""Plot a versioned scene gallery and accuracy/work tradeoffs from a saved evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdtdmesh.data.schema import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evaluation / "report.json").read_text())
    _, scenes = read_manifest(args.evaluation / "manifest.json")
    families = {}
    for scene in scenes:
        families.setdefault(scene.family, scene)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    for ax, (family, spec) in zip(axes.ravel(), families.items()):
        sim = spec.build(spec.budgets[0])
        x = (np.arange(128) + 0.5) * sim.Lx / 128
        y = (np.arange(128) + 0.5) * sim.Ly / 128
        eps, _, _, pec = sim.sample(x, y, raster=True)
        display = np.where(pec, -1, eps)
        ax.imshow(display.T, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", vmin=-1, vmax=4)
        for lo, hi in ((0, 0.125), (0.875, 1)):
            ax.axvspan(lo, hi, color="gray", alpha=0.25)
            ax.axhspan(lo, hi, color="gray", alpha=0.25)
        for source in spec.sources:
            ax.plot(source["x"] / sim.Lx, source["y"] / sim.Ly, "k*", ms=7)
        for receiver in spec.receivers:
            ax.plot(receiver["x"] / sim.Lx, receiver["y"] / sim.Ly, "ko", ms=3)
        ax.set(title=family, xlabel="x / Lx", ylabel="y / Ly")
    for ax in axes.ravel()[len(families) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Procedural geometry families / gray collars fixed / stars sources / dots receivers"
    )
    fig.tight_layout()
    fig.savefig(args.evaluation / "scene_gallery.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    styles = {"uniform": ("#3274ad", "o"), "heuristic": ("#d65f32", "s"), "cnn": ("#558a46", "^")}
    for strategy, (color, marker) in styles.items():
        rows = [
            r for r in report["candidates"] if r["status"] == "ok" and r["strategy"] == strategy
        ]
        if not rows:
            continue
        label = (
            "CNN (untrained)"
            if strategy == "cnn" and report["checkpoint"]["untrained"]
            else strategy
        )
        error = [r["metrics"]["waveform_l2"] for r in rows]
        for ax, key in zip(axes, ("cell_updates", "gpu_ms")):
            ax.scatter(
                [r["diagnostics"][key] for r in rows],
                error,
                c=color,
                marker=marker,
                label=label,
                alpha=0.75,
            )
            ax.set(xscale="log", yscale="log", ylabel="Mean relative receiver waveform L2")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
    axes[0].set_xlabel("Nx x Ny x Nt")
    axes[1].set_xlabel("GPU stepping time (ms; single run)")
    accepted = sum(s["status"] == "converged" for s in report["scenes"])
    fig.suptitle(
        f"Accuracy versus work and runtime / accepted references: {accepted}/{len(report['scenes'])}"
    )
    fig.text(
        0.5,
        0.015,
        "Each point is one scene, budget and mesh. Nonconverged references are excluded. "
        "Random CNN weights do not demonstrate learned improvement.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(args.evaluation / f"accuracy_cost.{suffix}", dpi=160)
    plt.close(fig)
    print(args.evaluation.resolve())


if __name__ == "__main__":
    main()
