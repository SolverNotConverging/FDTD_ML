"""Plot current CNN meshes across several exact cell budgets."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from fdtdmesh.data.schema import read_manifest  # noqa: E402


def material_background(spec, samples=320):
    simulation = spec.build(spec.budgets[-1])
    x = (np.arange(samples) + 0.5) * simulation.Lx / samples
    y = (np.arange(samples) + 0.5) * simulation.Ly / samples
    epsilon, _, _, pec = simulation.sample(x, y, raster=True)
    return epsilon.T, pec.T


def draw_scene(ax, spec, simulation, epsilon, pec, budget):
    ax.imshow(
        np.ma.masked_where(pec, epsilon),
        origin="lower",
        extent=(0, 1, 0, 1),
        norm=LogNorm(1, max(1.01, float(epsilon.max()))),
        cmap="YlOrRd",
        aspect="auto",
    )
    ax.imshow(
        np.ma.masked_where(~pec, pec),
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.85,
        aspect="auto",
    )
    if simulation is None:
        ax.text(0.5, 0.5, "infeasible", ha="center", va="center", weight="bold")
    else:
        line_width = max(0.13, 0.5 * 32 / budget)
        ax.vlines(
            simulation.mesh.x / simulation.Lx,
            0,
            1,
            color="#17324d",
            lw=line_width,
            alpha=0.68,
        )
        ax.hlines(
            simulation.mesh.y / simulation.Ly,
            0,
            1,
            color="#17324d",
            lw=line_width,
            alpha=0.68,
        )
    for source in spec.sources:
        ax.plot(source["x"] / spec.domain[0], source["y"] / spec.domain[1], "b*", ms=7)
    for receiver in spec.receivers:
        ax.plot(receiver["x"] / spec.domain[0], receiver["y"] / spec.domain[1], "bo", ms=2.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x / Lx", ylabel="y / Ly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenes", nargs="+", default=["test_iid-00000", "test_iid-00004", "test_iid-00018"]
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 48, 64, 96])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--title", default="Selected 80%-physics CNN: exact-budget graded meshes")
    args = parser.parse_args()
    if any(budget < 1 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Budgets must be distinct positive integers")

    _, scenes = read_manifest(args.manifest)
    scene_map = {scene.scene_id: scene for scene in scenes}
    if not all(scene_id in scene_map for scene_id in args.scenes):
        raise ValueError("Every requested scene must exist in the manifest")

    figure, axes = plt.subplots(
        len(args.scenes),
        len(args.budgets),
        figsize=(4.1 * len(args.budgets), 3.5 * len(args.scenes)),
        squeeze=False,
        constrained_layout=True,
    )
    meshes = {}
    for row, scene_id in enumerate(args.scenes):
        spec = scene_map[scene_id]
        epsilon, pec = material_background(spec)
        for column, budget in enumerate(args.budgets):
            simulation = spec.build([budget, budget])
            try:
                simulation.mesh_with_model(args.checkpoint, device=args.device)
            except Exception as error:
                if type(error).__name__ != "MeshInfeasibleError":
                    raise
                simulation = None
            meshes[scene_id, budget] = simulation
            draw_scene(axes[row, column], spec, simulation, epsilon, pec, budget)
            axes[row, column].set_title(f"{scene_id} / {budget}x{budget}")
    figure.suptitle(args.title, fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    profile_output = args.output.with_name(f"{args.output.stem}_cell_widths{args.output.suffix}")
    figure, axes = plt.subplots(
        len(args.scenes),
        2,
        figsize=(11, 3 * len(args.scenes)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, scene_id in enumerate(args.scenes):
        spec = scene_map[scene_id]
        for column, (axis, length) in enumerate(zip(("x", "y"), spec.domain)):
            for budget in args.budgets:
                simulation = meshes[scene_id, budget]
                if simulation is None:
                    continue
                lines = getattr(simulation.mesh, axis)
                centers = (lines[:-1] + lines[1:]) / (2 * length)
                normalized_width = np.diff(lines) * budget / length
                axes[row, column].plot(
                    centers, normalized_width, lw=1.15, label=f"{budget}x{budget}"
                )
            axes[row, column].axhline(1, color="black", ls=":", lw=0.8)
            axes[row, column].set(
                title=f"{scene_id} / {axis}-axis",
                xlabel=f"{axis} / L{axis}",
                ylabel="cell width / uniform width",
                xlim=(0, 1),
            )
            axes[row, column].legend(ncol=2, fontsize=8)
    figure.suptitle("CNN cell-width profiles across exact budgets", fontsize=15)
    figure.savefig(profile_output, dpi=180)
    plt.close(figure)
    print(args.output, profile_output)


if __name__ == "__main__":
    main()
