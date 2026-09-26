"""Frozen curriculum manifests and restartable independent-GPU data workers."""

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import generate_compact_poc_scenes, generate_lineages
from .qualification_v2 import qualify_compact_reference, qualify_reference
from .scoring_v2 import rank_candidates
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes
from .teacher_optimizer_v2 import (
    COMPACT_TEACHER_NAMES,
    optimize_teacher,
    promote_optimized_candidates,
)

INCIDENCE_ANGLES = (0.0, float(np.pi / 3))
BUDGETS = (32, 48, 64, 96)
OPTIMIZED_TEACHER_NAMES = ("optimized_0", "optimized_1", "optimized_2")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def scoring_fingerprint():
    source_root = Path(__file__).parent
    return {
        "accuracy_objective": "L_complex + 0.25 * L_scattering_width",
        "complex_weight": 1.0,
        "scattering_width_weight": 0.25,
        "time_step_exponent": 0.05,
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in ("metrics.py", "scoring_v2.py", "candidates_v2.py")
        },
    }


def freeze_manifest(profile_path, output, *, seed=20260923):
    """Fix corpus size and geometry lineages after the measured resource gate."""
    profile = json.loads(Path(profile_path).read_text())
    count = profile.get("selected_lineages")
    if profile.get("gate") != "ready_for_bulk" or count not in (128, 256):
        raise ValueError("Measured profile does not admit a 128/256-lineage campaign")
    output = Path(output)
    profile_hash = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "lineage_count": count,
        "seed": seed,
        "source_profile_sha256": profile_hash,
        "frequencies_hz": FREQUENCIES,
        "incidence_angles": INCIDENCE_ANGLES,
        "budgets": BUDGETS,
        "scenes": generate_lineages(count, seed=seed),
        "numerical_source_hashes": numerical_source_hashes(),
    }
    manifest = json.loads(json.dumps(manifest))
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != manifest:
            raise ValueError("Frozen campaign manifest differs from requested values")
    else:
        atomic_json(output, manifest)
    return manifest


def compact_conditions(scenes, seed=20260923):
    """Make deterministic train, validation, and final-test condition maps."""
    rng = np.random.default_rng(seed)
    conditions_by_scene = {}
    validation_conditions = {}
    test_conditions = {}
    for index, scene in enumerate(scenes):
        lineage = scene["lineage_id"]
        if scene["split"] == "train":
            angle_index = int(rng.integers(len(INCIDENCE_ANGLES)))
            selected_budgets = sorted(rng.choice(BUDGETS, size=2, replace=False).tolist())
            conditions_by_scene[lineage] = [
                {"angle_index": angle_index, "cells": int(cells)} for cells in selected_budgets
            ]
        elif scene["split"] == "validation":
            angle_index = index % len(INCIDENCE_ANGLES)
            conditions = [{"angle_index": angle_index, "cells": 64}]
            conditions_by_scene[lineage] = conditions
            validation_conditions[lineage] = conditions
        else:
            conditions = [
                {"angle_index": angle_index, "cells": cells}
                for angle_index in range(len(INCIDENCE_ANGLES))
                for cells in BUDGETS
            ]
            test_conditions[lineage] = conditions
    return conditions_by_scene, validation_conditions, test_conditions


def freeze_compact_manifest(profile_path, output, *, seed=20260923):
    """Freeze the 64/16/24 proof-of-concept suite after its measured cost gate."""
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    estimated_hours = profile.get("estimated_compact_campaign_hours")
    search_calibration = profile.get("teacher_search_calibration")
    reference_policy = profile.get("reference_policy")
    if (
        profile.get("gate") != "ready_for_compact_campaign"
        or profile.get("seed") != seed
        or isinstance(estimated_hours, bool)
        or not isinstance(estimated_hours, (int, float))
        or not math.isfinite(estimated_hours)
        or estimated_hours <= 0
        or estimated_hours > 24
        or not isinstance(reference_policy, dict)
        or not reference_policy.get("spatial_levels")
        or not isinstance(search_calibration, dict)
        or search_calibration.get("method") != "96_evaluation_smooth_density_differential_evolution"
        or not isinstance(
            search_calibration.get("estimated_teacher_search_wall_hours"), (int, float)
        )
        or search_calibration["estimated_teacher_search_wall_hours"] <= 0
        or not profile.get("reference_probes_pass")
        or not profile.get("positive_development_teacher_gain")
        or profile.get("qualified_reference_count") != 8
        or profile.get("missing_cost_cells")
        or profile.get("numerical_source_hashes") != numerical_source_hashes()
        or profile.get("scoring_fingerprint") != scoring_fingerprint()
    ):
        raise ValueError(
            "A complete, calibrated eight-scene profile with positive teacher benefit and a "
            "measured compact campaign estimate of at most 24 hours is required"
        )
    scenes = generate_compact_poc_scenes(seed=seed)
    angles = INCIDENCE_ANGLES
    conditions_by_scene, validation_conditions, test_conditions = compact_conditions(scenes, seed)
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 3,
        "protocol": "compact_c8_c9_poc_v1",
        "lineage_count": 104,
        "split_counts": {"train": 64, "validation": 16, "test": 24},
        "seed": seed,
        "source_profile_sha256": profile_hash,
        "estimated_compact_campaign_hours": float(estimated_hours),
        "reference_policy": reference_policy,
        "reference_policy_source_profile_sha256": profile_hash,
        "frequencies_hz": FREQUENCIES,
        "incidence_angles": angles,
        "budgets": BUDGETS,
        "candidate_names": COMPACT_TEACHER_NAMES + OPTIMIZED_TEACHER_NAMES,
        "teacher_search": {
            "method": "seeded_differential_evolution_smooth_log_density_v1",
            "evaluations_per_condition": 96,
            "retained_optimized_profiles": len(OPTIMIZED_TEACHER_NAMES),
            "calibration": search_calibration,
            "optimizer_sha256": hashlib.sha256(
                (Path(__file__).parent / "teacher_optimizer_v2.py").read_bytes()
            ).hexdigest(),
        },
        "conditions_by_scene": conditions_by_scene,
        "evaluation_conditions": {
            "validation": validation_conditions,
            "test": test_conditions,
        },
        "scenes": scenes,
        "numerical_source_hashes": numerical_source_hashes(),
        "scoring_fingerprint": scoring_fingerprint(),
    }
    manifest = json.loads(json.dumps(manifest))
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError("Frozen compact campaign manifest differs from requested values")
    else:
        atomic_json(output, manifest)
    return manifest


def _conditions_for_scene(manifest, scene, split=None):
    mapping = manifest.get("conditions_by_scene", {})
    if scene["lineage_id"] in mapping:
        return mapping[scene["lineage_id"]]
    if manifest.get("protocol") in ("compact_c8_c9_poc_v1", "c0_restart_v1"):
        return []
    if split is not None and scene["split"] != split:
        return []
    return [
        {"angle_index": angle_index, "cells": cells}
        for angle_index in range(len(manifest["incidence_angles"]))
        for cells in manifest["budgets"]
    ]


def _read_axes(directory):
    with np.load(directory / "spectra.npz") as arrays:
        return arrays["x"].copy(), arrays["y"].copy()


def _incomplete_case(directory, scene, angle, cells, policy):
    path = directory / "record.json"
    if path.exists():
        return
    atomic_json(
        path,
        {
            "schema_version": 2,
            "scene_id": scene["lineage_id"],
            "incidence_angle_rad": angle,
            "cells_x": cells,
            "cells_y": cells,
            "policy": policy,
            "status": "incomplete",
            "accepted": False,
            "reason": "campaign_deadline",
        },
    )


def run_data_worker(manifest_path, output, *, shard, shards=4, device="cuda:0", deadline=None):
    """One scene stays on one worker, avoiding duplicate CUDA/reference writes."""
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= shard < shards:
        raise ValueError("Shard must be within the worker count")
    output = Path(output)
    source_hashes = numerical_source_hashes()
    if manifest["numerical_source_hashes"] != source_hashes:
        raise ValueError("Numerical implementation changed after dataset freeze")
    if manifest.get("protocol") in ("compact_c8_c9_poc_v1", "c0_restart_v1"):
        if manifest.get("scoring_fingerprint") != scoring_fingerprint():
            raise ValueError("Scoring implementation changed after compact campaign freeze")
        search_config = manifest.get("teacher_search")
        if (
            search_config
            and search_config["optimizer_sha256"]
            != hashlib.sha256(
                (Path(__file__).parent / "teacher_optimizer_v2.py").read_bytes()
            ).hexdigest()
        ):
            raise ValueError("Teacher optimizer changed after compact campaign freeze")
    summary = {
        "shard": shard,
        "device": device,
        "references_accepted": 0,
        "references_rejected": 0,
        "cases_accepted": 0,
        "cases_other": 0,
        "cases_incomplete": 0,
        "started_at": time.time(),
    }
    candidate_names = tuple(manifest.get("candidate_names", CANDIDATE_NAMES))
    for scene_index, scene in enumerate(manifest["scenes"]):
        if scene_index % shards != shard:
            continue
        conditions = _conditions_for_scene(manifest, scene)
        by_angle = {}
        for condition in conditions:
            by_angle.setdefault(condition["angle_index"], set()).add(condition["cells"])
        for angle_index, selected_budgets in sorted(by_angle.items()):
            angle = manifest["incidence_angles"][angle_index]
            if deadline is not None and time.time() >= deadline:
                summary["deadline_reached"] = True
                atomic_json(output / "workers" / f"shard_{shard}.json", summary)
                return summary
            if manifest.get("protocol") in ("compact_c8_c9_poc_v1", "c0_restart_v1"):
                qualified = qualify_compact_reference(
                    scene,
                    angle_index,
                    angle,
                    output,
                    manifest["reference_policy"],
                    calibration_profile_sha256=manifest["reference_policy_source_profile_sha256"],
                    device=device,
                    deadline=deadline,
                )
            else:
                qualified = qualify_reference(
                    scene, angle_index, angle, output, device=device, deadline=deadline
                )
            if not qualified.get("accepted"):
                summary["references_rejected"] += 1
                continue
            summary["references_accepted"] += 1
            reference_path = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.npz"
            with np.load(reference_path) as arrays:
                reference = arrays["complex_far_field"].copy()
            for cells in sorted(selected_budgets):
                rows, directories = [], {}
                seed_names = [
                    name
                    for name in candidate_names
                    if not name.startswith(("refine_", "optimized_"))
                ]
                refine_names = [name for name in candidate_names if name.startswith("refine_")]
                optimized_names = [
                    name for name in candidate_names if name.startswith("optimized_")
                ]
                for name in seed_names:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    directories[name] = directory
                    if deadline is not None and time.time() >= deadline:
                        _incomplete_case(directory, scene, angle, cells, name)
                        summary["cases_incomplete"] += 1
                        continue
                    record, _ = evaluate_case(
                        scene,
                        cells,
                        angle,
                        name,
                        directory,
                        device=device,
                        reference=reference,
                        source_hashes=source_hashes,
                    )
                    rows.append({"name": name, **record})
                    summary["cases_accepted" if record.get("accepted") else "cases_other"] += 1
                if optimized_names:
                    search_output = (
                        output / "teacher_search" / f"{scene['lineage_id']}_a{angle_index}_n{cells}"
                    )
                    slots = (
                        output
                        / "teacher_promotions"
                        / f"{scene['lineage_id']}_a{angle_index}_n{cells}"
                    )
                    uniform = next((row for row in rows if row["name"] == "uniform"), None)
                    if (
                        uniform
                        and uniform.get("accepted")
                        and (deadline is None or time.time() < deadline)
                    ):
                        search_seed = int.from_bytes(
                            hashlib.sha256(
                                f"{scene['lineage_id']}:{angle_index}:{cells}".encode()
                            ).digest()[:4],
                            "little",
                        )
                        report = optimize_teacher(
                            scene,
                            cells,
                            angle,
                            reference,
                            search_output,
                            device=device,
                            evaluations=manifest["teacher_search"]["evaluations_per_condition"],
                            seed=search_seed,
                            uniform_record=uniform,
                            reference_uncertainty=qualified.get("reference_uncertainty_relative"),
                            deadline=deadline,
                        )
                        promotions = promote_optimized_candidates(
                            report, search_output, slots, count=len(optimized_names)
                        )
                    else:
                        promotions = [
                            {
                                "name": name,
                                "status": "incomplete"
                                if deadline is not None and time.time() >= deadline
                                else "no_valid_uniform_seed",
                            }
                            for name in optimized_names
                        ]
                        for item in promotions:
                            atomic_json(
                                slots / item["name"] / "record.json", item | {"accepted": False}
                            )
                    for item in promotions:
                        name = item["name"]
                        source = slots / name
                        destination = (
                            output
                            / "cases"
                            / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        for filename in ("record.json", "spectra.npz"):
                            if (source / filename).exists():
                                shutil.copy2(source / filename, destination / filename)
                        summary[
                            "cases_accepted" if item["status"] == "accepted" else "cases_other"
                        ] += 1
                        if item["status"] == "incomplete":
                            summary["cases_incomplete"] += 1
                try:
                    best = rank_candidates(rows)[0]
                    best_axes = _read_axes(directories[best["name"]])
                except (ValueError, IndexError, KeyError, OSError):
                    best_axes = None
                for name in refine_names:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    if deadline is not None and time.time() >= deadline:
                        _incomplete_case(directory, scene, angle, cells, name)
                        summary["cases_incomplete"] += 1
                        continue
                    if best_axes is None:
                        atomic_json(
                            directory / "record.json",
                            {
                                "schema_version": 2,
                                "scene_id": scene["lineage_id"],
                                "cells_x": cells,
                                "cells_y": cells,
                                "policy": name,
                                "status": "no_valid_seed",
                                "accepted": False,
                            },
                        )
                        summary["cases_other"] += 1
                        continue
                    record, _ = evaluate_case(
                        scene,
                        cells,
                        angle,
                        name,
                        directory,
                        device=device,
                        reference=reference,
                        selected_axes=best_axes,
                        source_hashes=source_hashes,
                    )
                    summary["cases_accepted" if record.get("accepted") else "cases_other"] += 1
            atomic_json(output / "workers" / f"shard_{shard}.json", summary)
    summary["finished_at"] = time.time()
    atomic_json(output / "workers" / f"shard_{shard}.json", summary)
    return summary


def mark_unfinished(manifest_path, output):
    """Materialize deadline-interrupted work as resumable incomplete records."""
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    counts = {"references": 0, "cases": 0}
    candidate_names = tuple(manifest.get("candidate_names", CANDIDATE_NAMES))
    for scene in manifest["scenes"]:
        conditions = _conditions_for_scene(manifest, scene)
        by_angle = {}
        for condition in conditions:
            by_angle.setdefault(condition["angle_index"], set()).add(condition["cells"])
        for angle_index, selected_budgets in by_angle.items():
            angle = manifest["incidence_angles"][angle_index]
            qualification = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
            if not qualification.exists():
                atomic_json(
                    qualification,
                    {
                        "schema_version": 2,
                        "terminal": False,
                        "accepted": False,
                        "status": "incomplete",
                        "reason": "campaign_deadline",
                        "scene_id": scene["lineage_id"],
                        "angle_index": angle_index,
                    },
                )
                counts["references"] += 1
            for cells in selected_budgets:
                for name in candidate_names:
                    directory = (
                        output / "cases" / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
                    )
                    if not (directory / "record.json").exists():
                        _incomplete_case(directory, scene, angle, cells, name)
                        counts["cases"] += 1
    atomic_json(output / "incomplete_counts.json", counts)
    return counts
