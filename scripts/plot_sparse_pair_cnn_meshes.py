#!/usr/bin/env python3
"""Plot pair-CNN meshes for representative two-object shape/material pairings."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from scattermesh.distillation import conditioning_features_v2, probability_axis, rasterize_scene
from scattermesh.model import AxisDensityUNet
from scattermesh.sparse import sparse_cluster_metrics

ROOT = Path(__file__).resolve().parents[1]
PAIR_COLUMNS = (
    ("circle+circle", "dielectric+dielectric", "Dielectric circles"),
    ("circle+rectangle", "dielectric+dielectric", "Dielectric circle + rectangle"),
    ("circle+rectangle", "dielectric+pec", "Dielectric circle + PEC rectangle"),
    ("rectangle+rectangle", "pec+pec", "PEC rectangles"),
)


def _object_center(obj):
    if obj["shape"] == "circle":
        return obj["center_m"]
    left, right, bottom, top = obj["bounds_m"]
    return ((left + right) / 2, (bottom + top) / 2)


def _object_bounds(obj):
    if obj["shape"] == "circle":
        x, y = obj["center_m"]
        radius = float(obj["radius_m"])
        return x - radius, x + radius, y - radius, y + radius
    return tuple(map(float, obj["bounds_m"]))


def _pair_examples(dataset_path, cluster_config_path):
    dataset = json.loads(dataset_path.read_text())
    test_pairs = [
        row for row in dataset["examples"]
        if row.get("family") == "sparse_pair" and row.get("split") == "test"
        and int(row["cells_x"]) == 32
    ]
    selected = []
    for shapes, materials, label in PAIR_COLUMNS:
        matches = [
            row for row in test_pairs
            if row["shape_topology"] == shapes and row["material_topology"] == materials
        ]
        if not matches:
            raise ValueError(f"No held-out pair example for {label}")
        example = dict(sorted(matches, key=lambda row: row["sample_id"])[0])
        selected.append({"label": label, "example": example, "source": example["sample_id"]})

    # Add a two-object slice of a held-out multi-object scene as a clearly marked
    # circular-PEC probe. This geometry is not in the pair fine-tuning data.
    cluster_config = json.loads(cluster_config_path.read_text())
    pec_circle_scenes = [
        scene for scene in cluster_config["scenes"]
        if scene.get("family_id") == "three_mixed_circle_pec" and scene.get("split") == "test"
    ]
    if not pec_circle_scenes:
        raise ValueError("No held-out one-circle-PEC cluster scene is available")
    scene = sorted(pec_circle_scenes, key=lambda row: row["scene_id"])[0]
    circles = [obj for obj in scene["objects"]
               if obj["shape"] == "circle" and obj["material"].get("kind") == "pec"]
    dielectrics = [obj for obj in scene["objects"]
                   if obj["material"].get("kind") != "pec"]
    pec = circles[0]
    dielectric = min(
        dielectrics,
        key=lambda obj: np.linalg.norm(
            np.asarray(_object_center(obj), dtype=float)
            - np.asarray(_object_center(pec), dtype=float)
        ),
    )
    objects = [dielectric, pec]
    metrics = sparse_cluster_metrics({"objects": objects}, domain=cluster_config["domain_m"])
    sizes = []
    for obj in objects:
        if obj["shape"] == "circle":
            sizes.append(2 * float(obj["radius_m"]))
        else:
            left, right, bottom, top = map(float, obj["bounds_m"])
            sizes.append(min(right - left, top - bottom))
    probe = {
        "objects": objects,
        "feature_size_m": min(min(sizes), metrics["minimum_gap_m"]),
        "incidence_angle_rad": scene["incidence_angle_rad"],
        "frequencies_hz": cluster_config["frequencies_hz"],
        "cells_x": 32,
        "cells_y": 32,
    }
    selected.append({
        "label": "Dielectric + circular PEC (unseen pair probe)",
        "example": probe,
        "source": scene["scene_id"],
        "out_of_pair_training": True,
    })
    return selected, dataset


def _mesh(example, model, resolution, cells):
    item = dict(example)
    item["cells_x"] = item["cells_y"] = cells
    raster = torch.from_numpy(rasterize_scene(item, resolution))[None]
    condition = torch.from_numpy(conditioning_features_v2(item, max_ratio=3.0))[None]
    with torch.no_grad():
        profiles = model(raster, condition)[0].cpu().numpy()
    x, x_repair = probability_axis(profiles[0], cells, max_ratio=3.0)
    y, y_repair = probability_axis(profiles[1], cells, max_ratio=3.0)
    return x, y, x_repair, y_repair


def _draw_case(axis, example, x, y, label, budget):
    for obj in example["objects"]:
        material = obj["material"]
        if material.get("kind") == "pec":
            color, alpha, text_color = "#20242b", 0.94, "white"
            text = "PEC"
        else:
            epsilon = float(material["epsilon_r"])
            color = plt.cm.viridis(np.log(epsilon) / np.log(30.0))
            alpha, text_color = 0.24, "#111827"
            text = rf"$\epsilon_r={epsilon:g}$"
        if obj["shape"] == "circle":
            patch = Circle(
                obj["center_m"], obj["radius_m"], facecolor=color,
                edgecolor="#111827", linewidth=1.25, alpha=alpha, zorder=2,
            )
        else:
            left, right, bottom, top = map(float, obj["bounds_m"])
            patch = Rectangle(
                (left, bottom), right - left, top - bottom,
                facecolor=color, edgecolor="#111827", linewidth=1.25,
                alpha=alpha, zorder=2,
            )
        axis.add_patch(patch)
        axis.text(
            *_object_center(obj), text, ha="center", va="center", color=text_color,
            fontsize=7, weight="bold", zorder=4,
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none", "pad": 1.0},
        )

    extent = [_object_bounds(obj) for obj in example["objects"]]
    left = min(bounds[0] for bounds in extent)
    right = max(bounds[1] for bounds in extent)
    bottom = min(bounds[2] for bounds in extent)
    top = max(bounds[3] for bounds in extent)
    span = max(right - left, top - bottom)
    margin = max(0.055, 0.55 * span)
    xmin, xmax = max(0.0, left - margin), min(1.2, right + margin)
    ymin, ymax = max(0.0, bottom - margin), min(1.2, top + margin)
    axis.vlines(x, ymin, ymax, color="#2457a7", linewidth=0.53, alpha=0.68, zorder=3)
    axis.hlines(y, xmin, xmax, color="#b43c35", linewidth=0.53, alpha=0.68, zorder=3)
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    axis.set_xticks(np.linspace(xmin, xmax, 3), labels=[f"{v:.2f}" for v in np.linspace(xmin, xmax, 3)])
    axis.set_yticks(np.linspace(ymin, ymax, 3), labels=[f"{v:.2f}" for v in np.linspace(ymin, ymax, 3)])
    axis.tick_params(labelsize=6, length=2)
    axis.grid(color="#cbd5e1", linewidth=0.35, alpha=0.7, zorder=1)
    axis.set_title(f"{budget} × {budget} cells", fontsize=8, pad=3)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "runs/sparse_nine_model_finetune_96/r128_c32/checkpoint.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "runs/sparse_joint_dataset_96/dataset.json",
    )
    parser.add_argument(
        "--cluster-config", type=Path, default=ROOT / "configs/sparse_cluster_pilot_32.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/sparse_pair_mesh_gallery/cnn_two_object_meshes.png",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 48, 64])
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    examples, dataset = _pair_examples(args.dataset, args.cluster_config)
    rows = []
    figure, axes = plt.subplots(
        len(args.budgets), len(examples),
        figsize=(3.35 * len(examples), 2.75 * len(args.budgets)),
        squeeze=False, constrained_layout=True,
    )
    with torch.no_grad():
        for column, case in enumerate(examples):
            for row, budget in enumerate(args.budgets):
                example = case["example"]
                x, y, x_repair, y_repair = _mesh(
                    example, model, checkpoint["config"]["raster_resolution"], budget
                )
                _draw_case(axes[row, column], example, x, y, case["label"], budget)
                if column == 0:
                    axes[row, column].set_ylabel("y (m)", fontsize=8)
                if row == len(args.budgets) - 1:
                    axes[row, column].set_xlabel("x (m)", fontsize=8)
                record = {
                    "label": case["label"],
                    "source": case["source"],
                    "budget": budget,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "x_uniform_repair_fraction": x_repair,
                    "y_uniform_repair_fraction": y_repair,
                    "out_of_pair_training": case.get("out_of_pair_training", False),
                    "objects": example["objects"],
                }
                rows.append(record)
        for column, case in enumerate(examples):
            axes[0, column].text(
                0.5, 1.31, case["label"], transform=axes[0, column].transAxes,
                ha="center", va="bottom", fontsize=9, weight="bold", wrap=True,
            )
    figure.suptitle(
        "Two-object meshes predicted by the pair-fine-tuned CNN\n"
        "Local views; each mesh uses the stated exact full-domain budget. "
        "Blue: x lines; red: y lines.",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp" + args.output.suffix)
    figure.savefig(temporary, dpi=190, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(args.output)
    metadata = {
        "schema_version": 1,
        "plot": str(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_status": json.loads(
            (args.checkpoint.parent / "summary.json").read_text()
        ).get("status"),
        "dataset_id": dataset.get("dataset_id"),
        "budgets": args.budgets,
        "description": "Pair-CNN two-object mesh examples; circular PEC panel is an unseen-family probe.",
        "cases": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"plot": str(args.output), "metadata": str(args.output.with_suffix('.json'))}))


if __name__ == "__main__":
    main()
