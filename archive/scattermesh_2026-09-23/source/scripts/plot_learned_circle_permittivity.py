#!/usr/bin/env python3
"""Plot low-budget meshes predicted for one circle across permittivity."""

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
REFERENCE_FREQUENCY_HZ = 1.0e9


def conductivity(epsilon_r, loss_tangent):
    return loss_tangent * 2 * np.pi * REFERENCE_FREQUENCY_HZ * EPS0 * epsilon_r


def grading_ratio(axis):
    spacing = np.diff(axis)
    return float(
        max(
            np.max(spacing[1:] / spacing[:-1]),
            np.max(spacing[:-1] / spacing[1:]),
        )
    )


def example(epsilon_r, cells, *, radius, center, angle, loss_tangent):
    return {
        "shape": "circle",
        "center_m": list(center),
        "radius_m": radius,
        "feature_size_m": 2 * radius,
        "epsilon_r": epsilon_r,
        "sigma_e_s_per_m": conductivity(epsilon_r, loss_tangent),
        "incidence_angle_rad": angle,
        "frequencies_hz": [0.8e9, 1.0e9, 1.2e9],
        "cells_x": cells,
        "cells_y": cells,
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mesh_distillation_pilot/training_v2/checkpoint.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/mesh_distillation_pilot/learned_circle_eps_low_budget.png"),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 48])
    parser.add_argument(
        "--epsilon-r", type=float, nargs="+", default=[2, 4, 8, 12, 20, 30]
    )
    parser.add_argument("--loss-tangent", type=float, default=0.10)
    parser.add_argument("--radius", type=float, default=0.085)
    parser.add_argument("--center", type=float, nargs=2, default=[0.58, 0.62])
    parser.add_argument("--angle", type=float, default=0.7)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    args = parser.parse_args()
    if min(args.budgets) < 4:
        raise ValueError("Budgets must contain at least four cells per axis")
    if not args.epsilon_r or min(args.epsilon_r) <= 1 or max(args.epsilon_r) > 30:
        raise ValueError("epsilon-r must lie in (1, 30]")
    if args.loss_tangent < 0:
        raise ValueError("loss-tangent must be nonnegative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    with torch.no_grad():
        for cells in args.budgets:
            for epsilon_r in args.epsilon_r:
                item = example(
                    epsilon_r,
                    cells,
                    radius=args.radius,
                    center=args.center,
                    angle=args.angle,
                    loss_tangent=args.loss_tangent,
                )
                raster = torch.from_numpy(rasterize_circle(item, 128))[None]
                conditioning = torch.from_numpy(conditioning_features(item))[None]
                profiles = model(raster, conditioning)[0].numpy()
                x, x_repair = probability_axis(
                    profiles[0], cells, max_ratio=args.max_ratio
                )
                y, y_repair = probability_axis(
                    profiles[1], cells, max_ratio=args.max_ratio
                )
                rows.append(
                    {
                        **item,
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "minimum_dx_m": float(np.diff(x).min()),
                        "maximum_dx_m": float(np.diff(x).max()),
                        "minimum_dy_m": float(np.diff(y).min()),
                        "maximum_dy_m": float(np.diff(y).max()),
                        "x_grading_ratio": grading_ratio(x),
                        "y_grading_ratio": grading_ratio(y),
                        "x_uniform_repair_fraction": x_repair,
                        "y_uniform_repair_fraction": y_repair,
                        "axis_hash": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                    }
                )

    figure, axes = plt.subplots(
        len(args.budgets),
        len(args.epsilon_r),
        figsize=(3.0 * len(args.epsilon_r), 3.0 * len(args.budgets)),
        squeeze=False,
        constrained_layout=True,
    )
    color_map = plt.get_cmap("viridis")
    by_key = {(row["cells_x"], row["epsilon_r"]): row for row in rows}
    for row_index, cells in enumerate(args.budgets):
        for column_index, epsilon_r in enumerate(args.epsilon_r):
            axis = axes[row_index, column_index]
            row = by_key[(cells, epsilon_r)]
            material_color = color_map(np.log(epsilon_r) / np.log(30))
            axis.add_patch(
                CirclePatch(
                    args.center,
                    args.radius,
                    facecolor=material_color,
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.72,
                    zorder=1,
                )
            )
            axis.vlines(row["x"], 0, DOMAIN, color="#2457a7", linewidth=0.42, alpha=0.62)
            axis.hlines(row["y"], 0, DOMAIN, color="#b43c35", linewidth=0.42, alpha=0.62)
            axis.arrow(
                0.08,
                0.10,
                0.10 * np.cos(args.angle),
                0.10 * np.sin(args.angle),
                width=0.004,
                head_width=0.025,
                color="black",
                length_includes_head=True,
                zorder=3,
            )
            axis.set(
                xlim=(0, DOMAIN),
                ylim=(0, DOMAIN),
                aspect="equal",
                xticks=[],
                yticks=[],
                title=(
                    rf"$\epsilon_r={epsilon_r:g}$, $\sigma_e={row['sigma_e_s_per_m']:.3f}$ S/m"
                    + "\n"
                    + rf"min $\Delta={1e3 * min(row['minimum_dx_m'], row['minimum_dy_m']):.1f}$ mm; "
                    + rf"grade $={max(row['x_grading_ratio'], row['y_grading_ratio']):.2f}$"
                ),
            )
            if column_index == 0:
                axis.set_ylabel(f"CNN mesh\n{cells} × {cells}", fontsize=11)
    figure.suptitle(
        "Pilot CNN low-budget meshes for one lossy dielectric circle\n"
        + rf"fixed radius={args.radius:.3f} m, center=({args.center[0]:.2f}, {args.center[1]:.2f}) m, "
        + rf"$\tan\delta={args.loss_tangent:.2f}$ at 1 GHz",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp" + args.output.suffix)
    figure.savefig(temporary, dpi=180)
    plt.close(figure)
    os.replace(temporary, args.output)
    metadata_path = args.output.with_suffix(".json")
    atomic_json(
        metadata_path,
        {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "description": "Controlled pilot-CNN permittivity sweep at exact low budgets",
            "loss_tangent_at_1ghz": args.loss_tangent,
            "max_grading_ratio": args.max_ratio,
            "cases": rows,
            "plot": str(args.output),
        },
    )
    print(json.dumps({"plot": str(args.output), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
