#!/usr/bin/env python3
"""Plot representative candidate meshes used by the pre-CNN physics campaign."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_simple_candidate_campaign.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("candidate_campaign_plot", RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def select_case(cases, geometries, *, epsilon_r, cells, candidate):
    return next(
        row
        for row in cases
        if geometries[row["geometry_id"]]["epsilon_r"] == epsilon_r
        and geometries[row["geometry_id"]]["variant_index"] == 0
        and row["cells"] == cells
        and row["candidate"] == candidate
    )


def main():
    runner = load_runner()
    manifest_path = ROOT / "configs/simple_dielectric_pool.json"
    campaign_path = ROOT / "configs/simple_candidate_full.json"
    inputs = runner.load_inputs(manifest_path, campaign_path)
    cases = runner.case_definitions(*inputs)
    manifest = json.loads(manifest_path.read_text())
    geometries = {row["geometry_id"]: row for row in manifest["geometries"]}
    selections = [
        (2.0, 32, "uniform"),
        (2.0, 32, "region_strong"),
        (4.5, 48, "region_strong"),
        (5.0, 64, "region_strong"),
    ]
    selected = [
        select_case(cases, geometries, epsilon_r=eps, cells=cells, candidate=candidate)
        for eps, cells, candidate in selections
    ]

    figure, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for axis, config in zip(axes.flat, selected, strict=True):
        grid = runner.PILOT.make_grid(config)
        scene = config["scene"]
        epsilon_r = scene["material"]["epsilon_r"]
        axis.add_patch(
            CirclePatch(
                scene["center"],
                scene["radius"],
                facecolor="#f4a261",
                edgecolor="#9d3d18",
                linewidth=1.5,
                alpha=0.72,
                zorder=3,
            )
        )
        axis.vlines(grid.x, 0, runner.PILOT.DOMAIN, color="#355070", linewidth=0.35, alpha=0.7)
        axis.hlines(grid.y, 0, runner.PILOT.DOMAIN, color="#355070", linewidth=0.35, alpha=0.7)
        axis.set(
            xlim=(0, runner.PILOT.DOMAIN),
            ylim=(0, runner.PILOT.DOMAIN),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=(
                f"$\\epsilon_r$={epsilon_r:g}, {config['cells']}×{config['cells']}\n"
                f"{config['candidate']} (physics candidate)"
            ),
        )
    figure.suptitle(
        "Current mesh-search examples — CNN training has not started",
        fontsize=14,
        fontweight="bold",
    )
    output = ROOT / "runs/simple_candidate_pilot/candidate_mesh_examples.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
