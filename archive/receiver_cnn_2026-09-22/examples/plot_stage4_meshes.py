"""Compare projected heuristic-teacher and trained CNN meshes on held-out scenes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from fdtdmesh.data.schema import read_manifest
from fdtdmesh.evaluation.pipeline import heuristic_density


def choose_scenes(rows, count=4):
    ordered = sorted(rows, key=lambda row: row["teacher_agreement"]["waveform_l2"])
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    return [ordered[index] for index in dict.fromkeys(indices)]


def build_meshes(spec, budget, checkpoint, device):
    teacher = spec.build(budget)
    teacher.mesh_from_density(*heuristic_density(teacher, spec.raster_shape))
    learned = spec.build(budget)
    learned.mesh_with_model(checkpoint, device=device)
    return teacher, learned


def background(spec, samples=300):
    simulation = spec.build(spec.budgets[0])
    x = (np.arange(samples) + 0.5) * simulation.Lx / samples
    y = (np.arange(samples) + 0.5) * simulation.Ly / samples
    epsilon, _, _, pec = simulation.sample(x, y, raster=True)
    return epsilon.T, pec.T


def draw_mesh(ax, spec, simulation, epsilon, pec, title):
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
    ax.vlines(simulation.mesh.x / simulation.Lx, 0, 1, color="#17324d", lw=0.28, alpha=0.65)
    ax.hlines(simulation.mesh.y / simulation.Ly, 0, 1, color="#17324d", lw=0.28, alpha=0.65)
    for source in spec.sources:
        ax.plot(source["x"] / spec.domain[0], source["y"] / spec.domain[1], "b*", ms=6)
    for receiver in spec.receivers:
        ax.plot(receiver["x"] / spec.domain[0], receiver["y"] / spec.domain[1], "bo", ms=2.5)
    ax.set(title=title, xlim=(0, 1), ylim=(0, 1), xlabel="x / Lx", ylabel="y / Ly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _, scenes = read_manifest(args.manifest)
    scene_map = {scene.scene_id: scene for scene in scenes}
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = evaluation["physical_teacher_agreement"]
    if args.scenes:
        row_map = {row["scene_id"]: row for row in rows}
        if not all(scene_id in row_map for scene_id in args.scenes):
            raise ValueError("Every requested scene must exist in the evaluation report")
        rows = [row_map[scene_id] for scene_id in args.scenes]
    else:
        rows = choose_scenes(rows)
    if not rows:
        raise ValueError("No evaluated scenes selected")
    args.output.mkdir(parents=True, exist_ok=True)
    built = []
    for row in rows:
        spec = scene_map[row["scene_id"]]
        teacher, learned = build_meshes(spec, row["budget"], args.checkpoint, args.device)
        built.append((row, spec, teacher, learned, *background(spec)))

    fig, axes = plt.subplots(len(built), 2, figsize=(10, 3.8 * len(built)), squeeze=False)
    for pair, (row, spec, teacher, learned, epsilon, pec) in zip(axes, built):
        errors = row["teacher_agreement"]
        draw_mesh(pair[0], spec, teacher, epsilon, pec, f"{spec.scene_id}: teacher")
        draw_mesh(
            pair[1],
            spec,
            learned,
            epsilon,
            pec,
            f"trained CNN / waveform {errors['waveform_l2']:.2%} / DFT {errors['spectrum_l2']:.2%}",
        )
    fig.suptitle("Held-out meshes: projected heuristic teacher vs trained CNN")
    fig.tight_layout()
    fig.savefig(args.output / "teacher_vs_cnn_meshes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(built), 2, figsize=(11, 2.7 * len(built)), squeeze=False)
    for pair, (row, spec, teacher, learned, _, _) in zip(axes, built):
        for ax, axis, length in zip(pair, ("x", "y"), spec.domain):
            for simulation, label, style in (
                (teacher, "teacher", "-"),
                (learned, "trained CNN", "--"),
            ):
                lines = getattr(simulation.mesh, axis)
                widths = np.diff(lines) / length
                centers = (lines[:-1] + lines[1:]) / (2 * length)
                ax.plot(centers, widths, style, lw=1.3, label=label)
            ax.set(
                title=f"{row['scene_id']} / {axis}-axis",
                xlabel=f"{axis} / L{axis}",
                ylabel=f"d{axis} / L{axis}",
                xlim=(0, 1),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Cell-width profiles reveal differences hidden by dense line overlays")
    fig.tight_layout()
    fig.savefig(args.output / "teacher_vs_cnn_cell_widths.png", dpi=180)
    plt.close(fig)
    print([row["scene_id"] for row in rows])


if __name__ == "__main__":
    main()
