#!/usr/bin/env python3
"""Draw one localized three/four-object geometry from each pilot family."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from scattermesh.sparse import sparse_cluster_metrics

ROOT = Path(__file__).resolve().parents[1]


def _center(obj):
    if obj["shape"] == "circle":
        return obj["center_m"]
    left, right, bottom, top = obj["bounds_m"]
    return ((left + right) / 2, (bottom + top) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_cluster_pilot_32.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_cluster_gallery.png")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    families = sorted({scene["family_id"] for scene in config["scenes"]})
    columns = min(3, len(families))
    rows = math.ceil(len(families) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows),
                            constrained_layout=True, squeeze=False)
    for axis, family in zip(axes.flat, families):
        scenes = [
            scene for scene in config["scenes"]
            if scene["family_id"] == family and scene["gap_stratum"] == "close"
        ]
        scene = scenes[0]
        metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
        for obj in scene["objects"]:
            material = obj["material"]
            pec = material["kind"] == "pec"
            color = "#292c35" if pec else plt.cm.viridis(
                min(float(material["epsilon_r"]) / 30, 1)
            )
            if obj["shape"] == "circle":
                patch = Circle(obj["center_m"], obj["radius_m"], facecolor=color,
                               edgecolor="white", linewidth=1.2)
            else:
                left, right, bottom, top = obj["bounds_m"]
                patch = Rectangle((left, bottom), right - left, top - bottom,
                                  facecolor=color, edgecolor="white", linewidth=1.2)
            axis.add_patch(patch)
            label = "PEC" if pec else f"ε={material['epsilon_r']:g}"
            axis.text(*_center(obj), label, color="white", fontsize=9,
                      ha="center", va="center", weight="bold")
        left, right, bottom, top = metrics["cluster_envelope_m"]
        axis.set_xlim(max(0, left - 0.05), min(config["domain_m"], right + 0.05))
        axis.set_ylim(max(0, bottom - 0.05), min(config["domain_m"], top + 0.05))
        axis.set_aspect("equal")
        axis.grid(color="#e5e7eb", linewidth=0.7)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(
            f"{family.replace('_', ' ')}  |  {scene['split']}\n"
            f"minimum gap {metrics['minimum_gap_m'] * 1000:.1f} mm, "
            f"incidence {scene['incidence_angle_rad'] * 180 / 3.141592653589793:.0f}°"
        )
    for axis in list(axes.flat)[len(families):]:
        axis.set_visible(False)
    fig.suptitle("Preflighted sparse multi-object pilot examples", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
