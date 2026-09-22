"""Plot meshes inferred by the distilled CNN on sparse and complex training scenes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
import torch

from fdtdmesh.data.schema import SceneSpec, digest
from fdtdmesh.ml import load_model
from fdtdmesh.physics import _predict_density

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/physics_distillation_mixed_v7_pml_v2"
OUTPUT = CAMPAIGN / "mesh_examples"
EXAMPLES = [
    ("train-v7-000000", "Single dielectric circle"),
    ("train-v7-000001", "Two dielectrics with a gap"),
    ("train-v7-000002", "Single PEC rectangle"),
    ("train-v7-000003", "PEC rectangle + dielectric"),
    ("train-v6-000002", "Complex: overlapping objects"),
    ("train-v6-000003", "Complex: nested objects + PEC wires"),
]


def draw(ax, spec, simulation, title):
    mesh = simulation.mesh
    lx, ly = np.array(spec.domain) * 1000
    coords = [(np.arange(700) + 0.5) * length / 700 for length in spec.domain]
    eps, _, _, pec = simulation.sample(*coords)
    cmap = LinearSegmentedColormap.from_list("dielectric", ["#ffffff", "#a9d8ee", "#ffc766", "#e36b40"])
    cmap.set_bad("#141923")
    im = ax.imshow(np.ma.masked_where(pec.T, eps.T), origin="lower",
                   extent=[0, lx, 0, ly], cmap=cmap, vmin=1, vmax=30, interpolation="nearest")
    x0, x1, y0, y1 = np.array(simulation.pml.interfaces(simulation)) * 1000
    for x, y, w, h in [(0, 0, x0, ly), (x1, 0, lx-x1, ly),
                        (x0, 0, x1-x0, y0), (x0, y1, x1-x0, ly-y1)]:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#bac3d0", alpha=.24, edgecolor="none"))
    lines = [[(x*1000, 0), (x*1000, ly)] for x in mesh.x]
    lines += [[(0, y*1000), (lx, y*1000)] for y in mesh.y]
    ax.add_collection(LineCollection(lines, colors="#334b61", linewidths=.43, alpha=.60))
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                          edgecolor="#586475", linestyle="--", linewidth=1.1))
    # Low-permittivity sparse objects need an outline to remain visible on the
    # shared 1–30 colour scale. Complex scenes retain the raster's overlap order.
    if len(spec.geometry) <= 2:
        for kind, material, data in simulation.primitives:
            if material.kind == "PEC":
                continue
            options = dict(fill=False, edgecolor="#526877", linewidth=1.0)
            if kind == "circle":
                x, y, radius = data
                ax.add_patch(Circle((x*1000, y*1000), radius*1000, **options))
            elif kind == "polygon":
                ax.add_patch(Polygon(data*1000, **options))
    for kind, material, data in simulation.primitives:
        if material.kind == "PEC" and kind == "line":
            x = [data.x]*2 if np.ndim(data.x) == 0 else data.x
            y = [data.y]*2 if np.ndim(data.y) == 0 else data.y
            ax.plot(np.array(x)*1000, np.array(y)*1000, color="#141923", lw=2.5)
    for probe, _ in simulation.sources:
        ax.plot(probe.x*1000, probe.y*1000, "*", color="#d82935", mec="white", mew=.8, ms=13)
    for probe in simulation.receivers:
        ax.plot(probe.x*1000, probe.y*1000, "x", color="#773fd4", mew=2, ms=8)
    ax.set(xlim=(0, lx), ylim=(0, ly), xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax.set_title(f"{title}\n{spec.scene_id} · {mesh.Nx} × {mesh.Ny} cells", fontsize=11, pad=9)
    ax.tick_params(labelsize=9)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=range(48, 129), default=64)
    args = parser.parse_args()
    budget = [args.budget, args.budget]
    output = OUTPUT if args.budget == 64 else CAMPAIGN / f"mesh_examples_{args.budget}x{args.budget}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    workflow = json.loads((CAMPAIGN / "workflow.json").read_text())
    manifest = json.loads(Path(workflow["manifest"]).read_text())
    assert digest(manifest["scenes"]) == workflow["dataset_id"]
    records = {r["scene_id"]: r for r in manifest["scenes"]}
    targets = json.loads((CAMPAIGN / "merged/physics_targets.json").read_text())
    trained = {(s["scene_id"], tuple(s["budget"])) for s in targets["samples"] if s["split"] == "train"}
    checkpoint = CAMPAIGN / "distillation/best.pt"
    model, metadata = load_model(checkpoint)
    output_records, scenes = [], []
    for sid, title in EXAMPLES:
        assert (sid, tuple(budget)) in trained
        spec = SceneSpec.from_dict(records[sid])
        density = _predict_density(spec, budget, model, metadata, "cpu")
        simulation = spec.build(budget)
        mesh = simulation.mesh_from_density(*density, time_limit=30)
        assert mesh.Nx == mesh.Ny == args.budget
        for axis in "xy":
            assert all(a in getattr(mesh, axis) for a in getattr(simulation, f"{axis}_anchors"))
        np.savez_compressed(output / f"{sid}.npz", x=mesh.x, y=mesh.y,
                            density_x=density[0], density_y=density[1])
        scenes.append((spec, simulation, title))
        output_records.append(dict(scene_id=sid, family=spec.family, split=spec.split,
                                   scene_hash=spec.content_hash, budget=budget,
                                   mesh_metadata=mesh.metadata))
        print(f"Inferred {sid}: {title}", flush=True)
    legend = [Line2D([], [], marker="*", linestyle="none", color="#d82935", markersize=11, label="Source"),
              Line2D([], [], marker="x", linestyle="none", color="#773fd4", markersize=8, label="Receiver"),
              Rectangle((0, 0), 1, 1, color="#141923", label="PEC"),
              Rectangle((0, 0), 1, 1, color="#dfe3ea", label="PML collar")]
    for filename, heading, selected, rows in (
        ("sparse_meshes", "Trained CNN meshes · sparse training scenes", scenes[:4], 2),
        ("complex_meshes", "Trained CNN meshes · complex training scenes", scenes[4:], 1),
    ):
        fig, axes = plt.subplots(rows, 2, figsize=(13.2, 6.1*rows+1), squeeze=False, layout="constrained")
        for ax, (spec, simulation, title) in zip(axes.flat, selected):
            im = draw(ax, spec, simulation, title)
        fig.suptitle(heading + f"\nPhysics-distilled checkpoint: epoch {metadata['training']['epoch']}", fontsize=17)
        fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
        fig.colorbar(im, ax=list(axes.flat), location="right", shrink=.8, fraction=.025,
                     label="Relative permittivity εr", ticks=[1, 10, 20, 30])
        fig.savefig(output / f"{filename}.png", dpi=180)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    (output / "examples.json").write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        epoch=metadata["training"]["epoch"], examples=output_records,
        note="New model predictions with constrained projection; not cached physics-search winners.",
    ), indent=2))
    print(output)


if __name__ == "__main__":
    main()
