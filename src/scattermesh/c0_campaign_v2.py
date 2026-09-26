"""Frozen C0 calibration and acquisition manifests for the compiled CUDA campaign."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .campaign_v2 import (
    BUDGETS,
    INCIDENCE_ANGLES,
    OPTIMIZED_TEACHER_NAMES,
    atomic_json,
    run_data_worker,
    scoring_fingerprint,
)
from .curriculum_v2 import scene_metrics
from .dataset_v2 import build_new_dataset
from .simulation_v2 import FREQUENCIES, numerical_source_hashes
from .teacher_optimizer_v2 import COMPACT_TEACHER_NAMES
from .training_v2 import train_model


REFERENCE_POLICY = {
    "protocol": "successive_uniform_c0_v1",
    "spatial_levels": [96, 128, 192, 256, 384],
    "complex_tolerance": 0.005,
    "width_tolerance": 0.01,
    "reference_tail_ratio": 1e-5,
    "pml_thickness_m": 0.12,
    "material_samples": 12,
    "durations_s": [7e-8, 1.4e-7, 5.6e-7],
}

ACQUISITION_REFERENCE_POLICY = REFERENCE_POLICY | {
    "protocol": "successive_uniform_c0_extended_v1",
    "spatial_levels": [96, 128, 192, 256, 384, 512, 768],
    "complex_tolerance": 0.01,
    "width_tolerance": 0.02,
}


def c0_scenes(count, *, seed=20260923, calibration=False):
    """Sample independent, bounded primitives with material strata and grouped splits."""
    if count < 12:
        raise ValueError("At least twelve C0 lineages are required")
    families = ("circle", "ellipse", "rectangle")
    rows = []
    for index in range(count):
        family = families[index % len(families)]
        rng = np.random.default_rng(seed + 7919 * index)
        center = rng.uniform(0.40, 0.80, size=2)
        width = float(rng.uniform(0.13, 0.31))
        height = float(rng.uniform(0.13, 0.31))
        # Four-way PEC cadence; lossless high-dielectric cases remain unchanged.
        if calibration:
            # Retain the first calibration panel exactly as frozen.
            material_class = (index // 3) % 6
            pec = material_class == 4
            epsilon_r = (2, 4, 8, 12, 8)[material_class if material_class < 4 else 4]
            lossy = material_class == 5
        else:
            # Acquisition targets approximately 75% dielectric and 25% PEC.
            pec = index % 4 == 3
            epsilon_r = (2, 4, 8, 12)[(index // 4 + index % 4) % 4]
            lossy = index % 12 == 8
        material = (
            {"kind": "pec"}
            if pec
            else {
                "kind": "dielectric",
                "epsilon_r": float(epsilon_r),
                "sigma_e_s_per_m": 0.01 if lossy else 0.0,
            }
        )
        common = {
            "schema_version": 2,
            "lineage_id": f"c0_{family}_{index:04d}",
            "family": family,
            "stage": "C0",
            "shape": family,
            "center_m": center.tolist(),
            "material": material,
        }
        if family == "circle":
            geometry = {"radius_m": float(np.sqrt(width * height) / 2)}
        elif family == "ellipse":
            geometry = {"radii_m": [width / 2, height / 2], "angle_rad": 0.0}
        else:
            geometry = {
                "bounds_m": [
                    float(center[0] - width / 2),
                    float(center[0] + width / 2),
                    float(center[1] - height / 2),
                    float(center[1] + height / 2),
                ]
            }
        scene = common | geometry
        metrics = scene_metrics(scene)
        if not 0.005 <= metrics["occupied_area_fraction"] <= 0.08:
            raise ValueError(f"C0 area out of bounds: {scene['lineage_id']}")
        if max(metrics["projected_x_support_fraction"], metrics["projected_y_support_fraction"]) > 0.35:
            raise ValueError(f"C0 projection out of bounds: {scene['lineage_id']}")
        if calibration:
            split = "train"
        else:
            split = "train"
        rows.append(scene | {"split": split})
    if not calibration:
        groups = {}
        for scene in rows:
            material = scene["material"]
            key = "pec" if material["kind"] == "pec" else f"epsilon_{int(material['epsilon_r'])}"
            groups.setdefault(key, []).append(scene)
        held_out = round(count / 8)
        quotas = {key: len(group) * held_out / count for key, group in groups.items()}
        allocations = {key: int(value) for key, value in quotas.items()}
        for key in sorted(groups, key=lambda name: (-(quotas[name] - allocations[name]), name))[:held_out - sum(allocations.values())]:
            allocations[key] += 1
        for key, group in groups.items():
            rng = np.random.default_rng(seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
            order = rng.permutation(len(group))
            held = allocations[key]
            if len(group) < 2 * held:
                raise ValueError(f"C0 material stratum too small for grouped split: {key}")
            for offset in order[:held]:
                group[int(offset)]["split"] = "validation"
            for offset in order[held:2 * held]:
                group[int(offset)]["split"] = "test"
    return rows


def make_manifest(output, *, count, seed=20260923, calibration=False, evaluations=96):
    output = Path(output)
    scenes = c0_scenes(count, seed=seed, calibration=calibration)
    rng = np.random.default_rng(seed + 11)
    conditions = {}
    for index, scene in enumerate(scenes):
        if calibration:
            # Thirty-six conditions for the default eighteen-scene panel.
            budgets = (BUDGETS[index % 4], BUDGETS[(index + 2) % 4])
            conditions[scene["lineage_id"]] = [
                {"angle_index": index % 2, "cells": cells} for cells in budgets
            ]
        elif scene["split"] != "test":
            angle = int(rng.integers(2))
            conditions[scene["lineage_id"]] = [
                {"angle_index": angle, "cells": cells} for cells in BUDGETS
            ]
    reference_policy = REFERENCE_POLICY if calibration else ACQUISITION_REFERENCE_POLICY
    manifest = {
        "schema_version": 4,
        "protocol": "c0_restart_v1",
        "purpose": "calibration" if calibration else "training_acquisition",
        "seed": seed,
        "lineage_count": count,
        "split_counts": {name: sum(s["split"] == name for s in scenes) for name in ("train", "validation", "test")},
        "scenes": scenes,
        "conditions_by_scene": conditions,
        "frequencies_hz": FREQUENCIES,
        "incidence_angles": INCIDENCE_ANGLES,
        "budgets": BUDGETS,
        "candidate_names": COMPACT_TEACHER_NAMES + OPTIMIZED_TEACHER_NAMES,
        "reference_policy": reference_policy,
        "reference_policy_source_profile_sha256": hashlib.sha256(json.dumps(reference_policy, sort_keys=True).encode()).hexdigest(),
        "teacher_search": {
            "method": "seeded_differential_evolution_smooth_log_density_v1",
            "evaluations_per_condition": evaluations,
            "optimizer_sha256": hashlib.sha256((Path(__file__).parent / "teacher_optimizer_v2.py").read_bytes()).hexdigest(),
        },
        "scoring_fingerprint": scoring_fingerprint(),
        "numerical_source_hashes": numerical_source_hashes(),
    }
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError("Frozen C0 manifest differs from requested values")
    else:
        atomic_json(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("manifest")
    prepare.add_argument("--count", type=int, default=18)
    prepare.add_argument("--calibration", action="store_true")
    prepare.add_argument("--seed", type=int, default=20260923)
    prepare.add_argument("--evaluations", type=int, default=96)
    worker = sub.add_parser("worker")
    worker.add_argument("manifest")
    worker.add_argument("output")
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--shards", type=int, default=4)
    worker.add_argument("--device", required=True)
    worker.add_argument("--deadline", type=float)
    dataset = sub.add_parser("dataset")
    dataset.add_argument("manifest")
    dataset.add_argument("output")
    dataset.add_argument("destination")
    train = sub.add_parser("train")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--deadline", type=float)
    train.add_argument("--base-channels", type=int, default=64)
    args = parser.parse_args()
    if args.command == "prepare":
        result = make_manifest(args.manifest, count=args.count, seed=args.seed, calibration=args.calibration, evaluations=args.evaluations)
        print(json.dumps({"manifest": args.manifest, "split_counts": result["split_counts"], "conditions": sum(map(len, result["conditions_by_scene"].values()))}))
    elif args.command == "worker":
        print(json.dumps(run_data_worker(args.manifest, args.output, shard=args.shard, shards=args.shards, device=args.device, deadline=args.deadline)))
    elif args.command == "dataset":
        print(build_new_dataset(args.manifest, args.output, args.destination))
    else:
        print(json.dumps(train_model(args.dataset, None, args.output, base_channels=args.base_channels, device=args.device, deadline=args.deadline, max_epochs=80, curriculum_mode="c0_restart", legacy_fraction=0.0)))


if __name__ == "__main__":
    main()
