#!/usr/bin/env python3
"""Plot the seven sparse-scene CNN inputs around a close object pair."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scattermesh.distillation import rasterize_scene
from scattermesh.sparse import sparse_cluster_metrics

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (
    "Dielectric fill", "log permittivity", "Conductivity", "PEC fill",
    "Signed distance", "Interface proximity", "Pair proximity",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--scene-id")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_input_channels.png")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128, 256, 384])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    scenes = config["scenes"]
    scene = (
        next(scene for scene in scenes if scene["scene_id"] == args.scene_id)
        if args.scene_id
        else min(
            scenes,
            key=lambda scene: sparse_cluster_metrics(
                scene, domain=config["domain_m"]
            )["minimum_gap_m"],
        )
    )
    metrics = sparse_cluster_metrics(scene, domain=config["domain_m"])
    left, right, bottom, top = metrics["cluster_envelope_m"]
    padding = max(0.035, 0.25 * max(right - left, top - bottom))
    bounds = (left - padding, right + padding, bottom - padding, top + padding)
    fig, axes = plt.subplots(
        len(args.resolutions), len(CHANNELS),
        figsize=(21, 3.4 * len(args.resolutions)), squeeze=False,
        constrained_layout=True,
    )
    for row, resolution in enumerate(args.resolutions):
        if resolution < 16:
            raise ValueError("Raster resolution must be at least 16")
        raster = rasterize_scene(scene, resolution, domain=config["domain_m"])
        for column, label in enumerate(CHANNELS):
            image = raster[column]
            axis = axes[row, column]
            axis.imshow(
                image, origin="lower", extent=(0, config["domain_m"], 0, config["domain_m"]),
                cmap="coolwarm" if column == 4 else "viridis",
                vmin=-1 if column == 4 else 0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_xlim(bounds[:2])
            axis.set_ylim(bounds[2:])
            axis.set_aspect("equal")
            axis.set_title(f"{label}\n{resolution} px, gap={metrics['minimum_gap_m'] * resolution / config['domain_m']:.2f} px")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(f"CNN physical inputs: {scene['scene_id']} (gap {metrics['minimum_gap_m'] * 1000:.1f} mm)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
