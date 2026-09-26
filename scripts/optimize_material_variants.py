#!/usr/bin/env python3
"""Compare material-specific optimized teachers on unfrozen development geometry."""

import copy
import json
from pathlib import Path

import numpy as np

from scattermesh.campaign_v2 import atomic_json
from scattermesh.constants import EPS0
from scattermesh.curriculum_v2 import generate_development_scenes, scene_metrics
from scattermesh.scoring_v2 import relative_l2
from scattermesh.simulation_v2 import evaluate_case
from scattermesh.teacher_optimizer_v2 import optimize_teacher

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs_v2" / "teacher_material_loss_pilot_v1"
LOSS_TANGENT_AT_1_GHZ = 0.02
CASES = (
    ("dev_c8_ship_template_v6", 8.0),
    ("dev_c8_vehicle_template_v7", 12.0),
    ("dev_c3_scene_20260924", 8.0),
)


def main():
    scenes = {row["lineage_id"]: row for row in generate_development_scenes()}
    summary = []
    for scene_id, epsilon_r in CASES:
        scene = copy.deepcopy(scenes[scene_id])
        scene["lineage_id"] += f"_er{int(epsilon_r)}"
        lossless_material = {
            "kind": "dielectric",
            "epsilon_r": epsilon_r,
            "sigma_e_s_per_m": 0.0,
        }
        for definition in scene.get("objects", [scene]):
            definition["material"] = lossless_material.copy()
        lossless, _ = evaluate_case(
            scene,
            48,
            0.0,
            "uniform",
            OUTPUT / "lossless_gate" / scene["lineage_id"],
            device="cuda:0",
        )
        if lossless.get("accepted"):
            result = {
                "scene_id": scene["lineage_id"],
                "epsilon_r": epsilon_r,
                "lossless_status": "accepted",
                "loss_variant_added": False,
            }
            summary.append(result)
            atomic_json(OUTPUT / "conditional_summary.json", summary)
            print(json.dumps(result), flush=True)
            continue
        if lossless["status"] != "unsettled":
            result = {
                "scene_id": scene["lineage_id"],
                "epsilon_r": epsilon_r,
                "lossless_status": lossless["status"],
                "loss_variant_added": False,
            }
            summary.append(result)
            atomic_json(OUTPUT / "conditional_summary.json", summary)
            print(json.dumps(result), flush=True)
            continue
        scene["lineage_id"] += "_loss002"
        conductivity = LOSS_TANGENT_AT_1_GHZ * 2 * np.pi * 1e9 * EPS0 * epsilon_r
        material = {
            "kind": "dielectric",
            "epsilon_r": epsilon_r,
            "sigma_e_s_per_m": float(conductivity),
        }
        for definition in scene.get("objects", [scene]):
            definition["material"] = material.copy()
        scene_metrics(scene)
        output = OUTPUT / scene["lineage_id"]
        atomic_json(output / "scene.json", scene)
        reference_rows = []
        fields = {}
        for cells in (192, 256, 384):
            record, _ = evaluate_case(
                scene, cells, 0.0, "uniform", output / "references" / f"n{cells}", device="cuda:0"
            )
            reference_rows.append({"cells": cells, "status": record["status"]})
            if not record.get("accepted"):
                break
            with np.load(output / "references" / f"n{cells}" / "spectra.npz") as arrays:
                fields[cells] = arrays["complex_far_field"].copy()
            if cells >= 256:
                previous = fields[192 if cells == 256 else 256]
                reference_rows[-1]["complex_change"] = relative_l2(fields[cells], previous)
                reference_rows[-1]["width_change"] = relative_l2(
                    2 * np.pi * np.abs(fields[cells]) ** 2,
                    2 * np.pi * np.abs(previous) ** 2,
                )
        result = {
            "scene_id": scene["lineage_id"],
            "epsilon_r": epsilon_r,
            "conductivity_s_per_m": float(conductivity),
            "loss_tangent_at_1_ghz": LOSS_TANGENT_AT_1_GHZ,
            "lossless_status": lossless["status"],
            "loss_variant_added": True,
            "reference_rows": reference_rows,
            "reference_qualified": len(fields) == 3
            and reference_rows[-1]["complex_change"] <= 0.005
            and reference_rows[-1]["width_change"] <= 0.005,
        }
        if len(fields) == 3:
            reference = fields[384]
            uniform, _ = evaluate_case(
                scene,
                48,
                0.0,
                "uniform",
                output / "uniform_n48",
                device="cuda:0",
                reference=reference,
            )
            if uniform.get("accepted"):
                report = optimize_teacher(
                    scene,
                    48,
                    0.0,
                    reference,
                    output / "optimized_n48",
                    device="cuda:0",
                    evaluations=96,
                    uniform_record=uniform,
                    reference_uncertainty=max(
                        reference_rows[-1]["complex_change"],
                        reference_rows[-1]["width_change"],
                    ),
                )
                result["gain"] = report["uniform_over_best_raw_accuracy_ratio"]
                result["best_teacher"] = report["best_teacher"]
                result["uniform_loss"] = uniform["joint_scattering_loss"]
                result["best_loss"] = report["best_raw_accuracy_loss"]
            else:
                result["uniform_status"] = uniform["status"]
        summary.append(result)
        atomic_json(OUTPUT / "conditional_summary.json", summary)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
