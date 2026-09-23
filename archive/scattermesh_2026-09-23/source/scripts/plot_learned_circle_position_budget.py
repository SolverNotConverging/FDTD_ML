#!/usr/bin/env python3
"""Plot learned meshes across circle position, material, scale, and exact budget."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle as CirclePatch

from scattermesh.constants import EPS0
from scattermesh.distillation import conditioning_features, probability_axis, rasterize_circle
from scattermesh.model import AxisDensityUNet

DOMAIN = 1.2
CASES = (
    {"name": "small southwest", "center_m": [0.34, 0.36], "radius_m": 0.045, "epsilon_r": 2.0, "loss_tangent": 0.0, "angle": 0.35},
    {"name": "medium center", "center_m": [0.60, 0.60], "radius_m": 0.075, "epsilon_r": 6.0, "loss_tangent": 0.04, "angle": 1.65},
    {"name": "large northeast", "center_m": [0.82, 0.82], "radius_m": 0.120, "epsilon_r": 20.0, "loss_tangent": 0.10, "angle": 4.90},
    {"name": "small southeast", "center_m": [0.84, 0.36], "radius_m": 0.060, "epsilon_r": 30.0, "loss_tangent": 0.15, "angle": 5.55},
)


def _conductivity(epsilon_r, loss_tangent):
    return loss_tangent * 2 * np.pi * 1e9 * EPS0 * epsilon_r


def _grading(axis):
    spacing = np.diff(axis)
    return float(max(np.max(spacing[1:] / spacing[:-1]), np.max(spacing[:-1] / spacing[1:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 48, 64, 96])
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = []
    with torch.no_grad():
        for cells in args.budgets:
            for definition in CASES:
                item = {
                    "shape": "circle",
                    "center_m": definition["center_m"],
                    "radius_m": definition["radius_m"],
                    "feature_size_m": 2 * definition["radius_m"],
                    "epsilon_r": definition["epsilon_r"],
                    "sigma_e_s_per_m": _conductivity(
                        definition["epsilon_r"], definition["loss_tangent"]
                    ),
                    "incidence_angle_rad": definition["angle"],
                    "frequencies_hz": [0.8e9, 1.0e9, 1.2e9],
                    "cells_x": cells,
                    "cells_y": cells,
                }
                raster = torch.from_numpy(rasterize_circle(item, 128))[None]
                condition = torch.from_numpy(conditioning_features(item))[None]
                profiles = model(raster, condition)[0].numpy()
                x, x_repair = probability_axis(profiles[0], cells, max_ratio=3.0)
                y, y_repair = probability_axis(profiles[1], cells, max_ratio=3.0)
                rows.append(
                    {
                        "name": definition["name"],
                        **item,
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "minimum_spacing_m": float(min(np.diff(x).min(), np.diff(y).min())),
                        "maximum_grading_ratio": max(_grading(x), _grading(y)),
                        "x_uniform_repair_fraction": x_repair,
                        "y_uniform_repair_fraction": y_repair,
                    }
                )

    figure, axes = plt.subplots(
        len(args.budgets), len(CASES), figsize=(12.8, 3.0 * len(args.budgets)),
        squeeze=False, constrained_layout=True,
    )
    lookup = {(row["cells_x"], row["name"]): row for row in rows}
    color_map = plt.get_cmap("viridis")
    for row_index, cells in enumerate(args.budgets):
        for column_index, definition in enumerate(CASES):
            row = lookup[(cells, definition["name"])]
            axis = axes[row_index, column_index]
            axis.add_patch(
                CirclePatch(
                    row["center_m"], row["radius_m"],
                    facecolor=color_map(np.log(row["epsilon_r"]) / np.log(30.0)),
                    edgecolor="black", linewidth=1.1, alpha=0.75, zorder=1,
                )
            )
            axis.vlines(row["x"], 0, DOMAIN, color="#2457a7", linewidth=0.38, alpha=0.62)
            axis.hlines(row["y"], 0, DOMAIN, color="#b43c35", linewidth=0.38, alpha=0.62)
            angle = row["incidence_angle_rad"]
            axis.arrow(
                0.08, 0.10, 0.11 * np.cos(angle), 0.11 * np.sin(angle),
                width=0.004, head_width=0.026, color="black", length_includes_head=True,
                zorder=3,
            )
            axis.set(
                xlim=(0, DOMAIN), ylim=(0, DOMAIN), aspect="equal", xticks=[], yticks=[],
                title=(
                    f"{row['name']}\n"
                    + rf"$\epsilon_r={row['epsilon_r']:g}$, $a={row['radius_m']:.3f}$ m; "
                    + rf"min $\Delta={1e3 * row['minimum_spacing_m']:.1f}$ mm"
                ),
            )
            if column_index == 0:
                axis.set_ylabel(f"CNN mesh\n{cells} × {cells}", fontsize=11)
    figure.suptitle(
        "Qualified CNN meshes across circle position, scale, material, and exact budget",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp" + args.output.suffix)
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, args.output)
    metadata = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "description": "Qualified-CNN position/material/scale/budget mesh examples",
        "cases": rows,
        "plot": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"plot": str(args.output), "metadata": str(args.output.with_suffix('.json'))}))


if __name__ == "__main__":
    main()
