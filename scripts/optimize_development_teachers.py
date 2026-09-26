#!/usr/bin/env python3
"""Bounded compiled-CUDA teacher search on already profiled development scenes."""

import argparse
import json
from pathlib import Path

import numpy as np

from scattermesh.curriculum_v2 import generate_development_scenes
from scattermesh.teacher_optimizer_v2 import optimize_teacher

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "runs_v2/compact_poc/profile")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs_v2/teacher_optimization_pilot_v1"
    )
    parser.add_argument(
        "--scene", action="append", help="Development lineage ID; repeat to select several"
    )
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--evaluations", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    profile = json.loads((arguments.profile / "compact_profile.json").read_text())
    if profile["qualified_reference_count"] != 8:
        raise ValueError("The complete eight-scene reference profile is required")
    scene_lookup = {
        scene["lineage_id"]: scene for scene in generate_development_scenes(profile["seed"])
    }
    row_lookup = {row["scene_id"]: row for row in profile["rows"]}
    selected = arguments.scene or [
        "dev_c3_scene_20260924",
        "dev_c5_scene_20260924",
        "dev_c8_aircraft_template_v5",
        "dev_c8_ship_template_v6",
        "dev_c8_vehicle_template_v7",
    ]
    for scene_id in selected:
        scene = scene_lookup[scene_id]
        row = row_lookup[scene_id]
        reference_directory = (
            arguments.profile
            / "scenes"
            / scene_id
            / "references"
            / f"uniform_n{row['reference_cells']}"
        )
        with np.load(reference_directory / "spectra.npz") as arrays:
            reference_key = (
                "reference_complex_far_field"
                if "reference_complex_far_field" in arrays
                else "complex_far_field"
            )
            reference = arrays[reference_key].copy()
        uniform_directory = (
            arguments.profile / "scenes" / scene_id / "teachers" / f"n{arguments.cells}_uniform"
        )
        uniform_record = json.loads((uniform_directory / "record.json").read_text())
        report = optimize_teacher(
            scene,
            arguments.cells,
            0.0,
            reference,
            arguments.output / scene_id / f"n{arguments.cells}",
            device=arguments.device,
            evaluations=arguments.evaluations,
            uniform_record=uniform_record,
            reference_uncertainty=row["reference_uncertainty_relative"],
        )
        print(
            scene_id,
            f"accepted={report['accepted_evaluations']}/{report['completed_evaluations']}",
            f"raw_gain={report['uniform_over_best_raw_accuracy_ratio']:.3f}",
            f"best={report['best_raw_accuracy_candidate']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
