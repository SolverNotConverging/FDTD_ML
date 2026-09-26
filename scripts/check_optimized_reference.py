#!/usr/bin/env python3
"""Re-rank saved development meshes against independently finer uniform fields."""

import json
from pathlib import Path

import numpy as np

from scattermesh.campaign_v2 import atomic_json
from scattermesh.curriculum_v2 import generate_development_scenes
from scattermesh.metrics import scattering_loss
from scattermesh.scoring_v2 import relative_l2
from scattermesh.simulation_v2 import evaluate_case

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "runs_v2" / "compact_poc" / "profile"
SEARCH = ROOT / "runs_v2" / "teacher_optimization_pilot_v1"
OUTPUT = ROOT / "runs_v2" / "teacher_reference_check_v1"


def _field(directory):
    with np.load(directory / "spectra.npz") as arrays:
        return arrays["complex_far_field"].copy()


def main():
    scenes = {scene["lineage_id"]: scene for scene in generate_development_scenes()}
    results = []
    for scene_id in ("dev_c8_ship_template_v6", "dev_c8_vehicle_template_v7"):
        scene = scenes[scene_id]
        search = json.loads((SEARCH / scene_id / "n48" / "optimization_report.json").read_text())
        best_name = search["best_raw_accuracy_candidate"]
        if best_name == "uniform":
            continue
        best_index = int(best_name.split("_")[-1])
        uniform_field = _field(PROFILE / "scenes" / scene_id / "teachers" / "n48_uniform")
        best_field = _field(SEARCH / scene_id / "n48" / "trials" / f"trial_{best_index:04d}")
        reference_fields = {}
        for cells in (192, 256, 384):
            record, _ = evaluate_case(
                scene,
                cells,
                0.0,
                "uniform",
                OUTPUT / scene_id / "references" / f"n{cells}",
                device="cuda:1",
            )
            if not record.get("accepted"):
                break
            reference_fields[cells] = _field(OUTPUT / scene_id / "references" / f"n{cells}")
        comparisons = []
        for cells, reference in reference_fields.items():
            uniform_loss = scattering_loss(uniform_field, reference)["joint_scattering_loss"]
            best_loss = scattering_loss(best_field, reference)["joint_scattering_loss"]
            comparisons.append(
                {
                    "reference_cells": cells,
                    "uniform_loss": uniform_loss,
                    "optimized_loss": best_loss,
                    "uniform_over_optimized_ratio": uniform_loss / best_loss,
                    "complex_change_from_previous": relative_l2(
                        reference, reference_fields[cells - 64]
                    )
                    if cells == 256 and 192 in reference_fields
                    else relative_l2(reference, reference_fields[256])
                    if cells == 384 and 256 in reference_fields
                    else None,
                }
            )
        results.append({"scene_id": scene_id, "best_trial": best_index, "comparisons": comparisons})
        atomic_json(OUTPUT / "summary.json", results)
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
