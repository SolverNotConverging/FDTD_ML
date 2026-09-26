"""Bounded C6 reference and mesh qualification for holes and open cavities."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .c3_qualification_v2 import (
    DEFAULT_REFERENCE_LEVELS as C3_REFERENCE_LEVELS,
)
from .c3_qualification_v2 import (
    DURATIONS,
    MONITOR_BOUNDS,
    _atomic_json,
    _qualify_reference,
)
from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import DOMAIN, scene_metrics
from .scoring_v2 import soft_dt_score
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes
from .topology_v2 import open_cavity_smoke_scene, square_ring_smoke_scene

DEFAULT_BUDGETS = (48, 64, 96)
DEFAULT_POLICIES = ("uniform", "interface", "hybrid")
DEFAULT_REFERENCE_LEVELS = (*C3_REFERENCE_LEVELS, 512)
TOPOLOGY_SIGMA_E_S_PER_M = 0.002
TOPOLOGY_WALLS_M = (0.09, 0.03, 0.015)


def generate_c6_topology_scenes():
    """Create a wall-thickness sweep and a matched-scale open-cavity example."""
    scenes = [
        square_ring_smoke_scene(
            wall_thickness_m=thickness,
            sigma_e_s_per_m=TOPOLOGY_SIGMA_E_S_PER_M,
        )
        for thickness in TOPOLOGY_WALLS_M
    ]
    scenes.append(open_cavity_smoke_scene(sigma_e_s_per_m=TOPOLOGY_SIGMA_E_S_PER_M))
    for scene in scenes:
        scene["split"] = "development"
        scene["frequency_hz"] = 1e9
        scene["scene_metrics"] = scene_metrics(scene)
    return scenes


def run_c6_topology_qualification(
    output,
    *,
    reference_levels=DEFAULT_REFERENCE_LEVELS,
    budgets=DEFAULT_BUDGETS,
    policies=DEFAULT_POLICIES,
    device="cuda:0",
):
    """Qualify bounded dielectric topology cases with reference and mesh controls."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(value) for value in reference_levels)
    budgets = tuple(int(value) for value in budgets)
    policies = tuple(policies)
    supported_levels = (64, 96, 128, 192, 256, 384, 512)
    if (
        len(levels) < 2
        or tuple(sorted(set(levels))) != levels
        or any(value not in supported_levels for value in levels)
    ):
        raise ValueError("Reference levels must be at least two increasing budgets")
    if not budgets or any(value not in (32, 48, 64, 96, 128, 192) for value in budgets):
        raise ValueError("Unsupported C6 candidate budget")
    if "uniform" not in policies or any(value not in CANDIDATE_NAMES for value in policies):
        raise ValueError("C6 policies must include uniform and use supported candidates")

    source_hashes = numerical_source_hashes()
    source_files = (
        Path(__file__),
        Path(__file__).with_name("c3_qualification_v2.py"),
        Path(__file__).with_name("topology_v2.py"),
    )
    campaign_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    reference_engine_source = source_files[1].read_bytes()
    scene_rows = []
    for scene in generate_c6_topology_scenes():
        scene_root = output / "scenes" / scene["lineage_id"]
        scene_root.mkdir(parents=True, exist_ok=True)
        scene_hash = hashlib.sha256(
            json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        qualification_path = scene_root / "qualification.json"
        qualification = None
        reference = None
        reference_path = qualification_path.with_suffix(".npz")
        if qualification_path.exists():
            saved = json.loads(qualification_path.read_text())
            if (
                saved.get("scene_sha256") == scene_hash
                and saved.get("source_hashes") == source_hashes
                and saved.get("campaign_source_sha256") == campaign_hash
                and saved.get("reference_levels") == list(levels)
                and saved.get("durations_s") == list(DURATIONS)
                and saved.get("monitor_bounds_m") == list(MONITOR_BOUNDS)
                and (not saved["qualification"].get("accepted") or reference_path.exists())
            ):
                qualification = saved["qualification"]
                if qualification.get("accepted"):
                    with np.load(reference_path) as arrays:
                        reference = arrays["complex_far_field"].copy()
        if qualification is None:
            qualification, reference = _qualify_reference(
                scene,
                scene_root,
                levels=levels,
                device=device,
                monitor_bounds=MONITOR_BOUNDS,
                durations=DURATIONS,
            )
            if qualification.get("accepted"):
                temporary = reference_path.with_name(reference_path.stem + ".tmp.npz")
                np.savez_compressed(temporary, complex_far_field=reference)
                os.replace(temporary, reference_path)
            _atomic_json(
                qualification_path,
                {
                    "scene_sha256": scene_hash,
                    "source_hashes": source_hashes,
                    "campaign_source_sha256": campaign_hash,
                    "reference_engine_source_sha256": hashlib.sha256(
                        reference_engine_source
                    ).hexdigest(),
                    "reference_levels": list(levels),
                    "durations_s": list(DURATIONS),
                    "monitor_bounds_m": list(MONITOR_BOUNDS),
                    "qualification": qualification,
                },
            )

        candidates = []
        if qualification.get("accepted"):
            for cells in budgets:
                for policy in policies:
                    directory = scene_root / "meshes" / f"n{cells}_{policy}"
                    record, _ = evaluate_case(
                        scene,
                        cells,
                        0.0,
                        policy,
                        directory,
                        device=device,
                        reference=reference,
                        pml_thickness=0.12,
                        material_samples=24,
                        durations=DURATIONS,
                        tail_tolerance=1e-5,
                        monitor_bounds=MONITOR_BOUNDS,
                    )
                    candidates.append(
                        {
                            "cells": cells,
                            "policy": policy,
                            "status": record["status"],
                            "accepted": bool(record.get("accepted")),
                            "complex_relative_l2": record.get("complex_relative_l2"),
                            "width_relative_l2": record.get("width_relative_l2"),
                            "joint_scattering_loss": record.get("joint_scattering_loss"),
                            "dt": record.get("dt"),
                            "Nt": record.get("Nt"),
                            "wall_seconds": record.get("wall_seconds"),
                            "uniform_repair_fraction": record.get("uniform_repair_fraction"),
                            "x_grading": record.get("x_grading"),
                            "y_grading": record.get("y_grading"),
                        }
                    )

        scoring = {}
        threshold_summary = {}
        for cells in budgets:
            rows = [row for row in candidates if row["cells"] == cells]
            uniform = next((row for row in rows if row["policy"] == "uniform"), None)
            scores_by_exponent = {}
            if uniform and uniform["accepted"]:
                for exponent in (0.0, 0.02, 0.05, 0.1):
                    scores = {
                        row["policy"]: soft_dt_score(
                            row["joint_scattering_loss"],
                            row["dt"],
                            uniform["dt"],
                            exponent=exponent,
                        )
                        for row in rows
                        if row["accepted"] and row["joint_scattering_loss"] is not None
                    }
                    scores_by_exponent[str(exponent)] = {
                        "scores": scores,
                        "best_policy": min(scores, key=scores.get) if scores else None,
                    }
            scoring[str(cells)] = {
                "status": "scored" if scores_by_exponent else "uniform_unsettled_or_unqualified",
                "scores_by_dt_exponent": scores_by_exponent,
            }
        for target in (0.02, 0.05, 0.10):
            by_policy = {}
            for policy in policies:
                passing = [
                    row["cells"]
                    for row in candidates
                    if row["policy"] == policy
                    and row["accepted"]
                    and row["complex_relative_l2"] is not None
                    and row["width_relative_l2"] is not None
                    and row["complex_relative_l2"] <= target
                    and row["width_relative_l2"] <= target
                ]
                by_policy[policy] = min(passing) if passing else None
            threshold_summary[str(target)] = by_policy

        metrics = scene["scene_metrics"]
        scene_rows.append(
            {
                "scene_id": scene["lineage_id"],
                "scene": scene,
                "topology": scene["family"],
                "minimum_feature_size_m": metrics["feature_size_m"],
                "minimum_feature_pixels_at_512": metrics["feature_size_m"] / (DOMAIN / 512),
                "ring_wall_thickness_m": scene.get("ring_wall_thickness_m"),
                "occupied_area_m2": metrics["occupied_area_fraction"] * DOMAIN**2,
                "union_x_projection_fraction": metrics["projected_x_support_fraction"],
                "union_y_projection_fraction": metrics["projected_y_support_fraction"],
                "reference_qualification": qualification,
                "mesh_results": candidates,
                "scoring_sensitivity_by_cells": scoring,
                "minimum_cells_by_joint_error_target": threshold_summary,
            }
        )

    status_counts = Counter(row["status"] for scene in scene_rows for row in scene["mesh_results"])
    report = {
        "schema_version": 1,
        "campaign": "c6_topology_feature_sweep_v1",
        "device": device,
        "topology_conductivity_s_per_m": TOPOLOGY_SIGMA_E_S_PER_M,
        "reference_levels": levels,
        "budgets": budgets,
        "policies": policies,
        "frequencies_hz": FREQUENCIES,
        "incidence_angles_rad": [0.0],
        "monitor_bounds_m": MONITOR_BOUNDS,
        "pml_thickness_m": 0.12,
        "material_samples": 24,
        "durations_s": DURATIONS,
        "dt_penalty_exponents": (0.0, 0.02, 0.05, 0.1),
        "source_hashes": source_hashes,
        "campaign_source_sha256": campaign_hash,
        "reference_engine_source_sha256": hashlib.sha256(reference_engine_source).hexdigest(),
        "qualified_scene_count": sum(
            row["reference_qualification"].get("accepted", False) for row in scene_rows
        ),
        "scene_count": len(scene_rows),
        "candidate_case_count": sum(len(row["mesh_results"]) for row in scene_rows),
        "candidate_status_counts": dict(status_counts),
        "scenes": scene_rows,
    }
    _atomic_json(output / "c6_topology_qualification_report.json", report)
    return report
