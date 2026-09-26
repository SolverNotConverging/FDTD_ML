"""Controlled C5 qualification across object counts and spatial arrangements."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .c3_qualification_v2 import (
    DEFAULT_REFERENCE_LEVELS,
    DURATIONS,
    PROBE_TOLERANCE,
    SPATIAL_COMPLEX_TOLERANCE,
    SPATIAL_WIDTH_TOLERANCE,
    _atomic_json,
    _qualify_reference,
)
from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import DOMAIN, _object_area, _scaled_definition, scene_metrics
from .scoring_v2 import soft_dt_score
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes

DEFAULT_BUDGETS = (32, 48, 64, 96)
DEFAULT_POLICIES = (
    "uniform",
    "interface",
    "center",
    "wide",
    "smooth_0",
    "smooth_1",
    "hybrid",
)
MONITOR_BOUNDS = (0.22, 0.98, 0.22, 0.98)
TARGET_OCCUPIED_AREA_M2 = 0.0068


def _base_objects():
    center = np.asarray((0.60, 0.60))
    material = {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0}
    theta = np.arange(10) * np.pi / 5
    radii = np.where(np.arange(10) % 2 == 0, 0.024, 0.013)
    star_vertices = np.column_stack(
        (center[0] + radii * np.cos(theta), center[1] + radii * np.sin(theta))
    )
    return [
        {
            "object_id": "template_circle",
            "shape": "circle",
            "center_m": center.tolist(),
            "radius_m": 0.025,
            "feature_size_m": 0.05,
            "material": material.copy(),
        },
        {
            "object_id": "template_ellipse",
            "shape": "ellipse",
            "center_m": center.tolist(),
            "radii_m": [0.030, 0.020],
            "angle_rad": 0.25,
            "feature_size_m": 0.04,
            "material": material.copy(),
        },
        {
            "object_id": "template_rectangle",
            "shape": "rotated_rectangle",
            "center_m": center.tolist(),
            "width_m": 0.045,
            "height_m": 0.028,
            "angle_rad": 0.35,
            "feature_size_m": 0.028,
            "material": material.copy(),
        },
        {
            "object_id": "template_triangle",
            "shape": "triangle",
            "center_m": center.tolist(),
            "vertices_m": [
                [0.625, 0.600],
                [0.580, 0.620],
                [0.580, 0.580],
            ],
            "feature_size_m": 0.04,
            "material": material.copy(),
        },
        {
            "object_id": "template_star",
            "shape": "star",
            "center_m": center.tolist(),
            "vertices_m": star_vertices.tolist(),
            "feature_size_m": 0.026,
            "material": material.copy(),
        },
    ]


def _layout_centers(count, layout):
    centers = {
        (3, "compact"): ((0.54, 0.52), (0.66, 0.52), (0.60, 0.63)),
        (5, "compact"): (
            (0.54, 0.49),
            (0.66, 0.49),
            (0.60, 0.60),
            (0.54, 0.71),
            (0.66, 0.71),
        ),
        (3, "aligned"): ((0.45, 0.60), (0.60, 0.60), (0.75, 0.60)),
        (5, "aligned"): (
            (0.38, 0.60),
            (0.49, 0.60),
            (0.60, 0.60),
            (0.71, 0.60),
            (0.82, 0.60),
        ),
        (3, "dispersed"): ((0.38, 0.38), (0.82, 0.38), (0.60, 0.82)),
        (5, "dispersed"): (
            (0.38, 0.38),
            (0.82, 0.38),
            (0.60, 0.60),
            (0.38, 0.82),
            (0.82, 0.82),
        ),
    }
    if count in (3, 5):
        return centers[count, layout]
    if count not in (6, 8, 10) or layout not in ("compact", "aligned", "dispersed"):
        raise ValueError("C5 controlled layouts support 3, 5, 6, 8, or 10 objects")
    if layout == "compact":
        locations = [(x, y) for y in (0.46, 0.55, 0.64) for x in (0.46, 0.55, 0.64, 0.73)]
        return tuple(locations[:count])
    if layout == "aligned":
        return tuple((float(x), 0.60) for x in np.linspace(0.28, 0.92, count))
    locations = [(x, y) for y in (0.34, 0.60, 0.86) for x in (0.32, 0.50, 0.68, 0.86)]
    return tuple(locations[:count])


def generate_c5_controlled_scenes(object_counts=(3, 5)):
    """Build matched-area C5 scenes in compact/aligned/dispersed layouts."""
    counts = tuple(int(value) for value in object_counts)
    if (
        not counts
        or len(set(counts)) != len(counts)
        or any(value not in (3, 5, 6, 8, 10) for value in counts)
    ):
        raise ValueError("C5 object counts must be unique values from 3, 5, 6, 8, and 10")
    templates = _base_objects()
    template_areas = [_object_area_from_definition(item) for item in templates]
    area_by_count = {
        count: sum(template_areas[index % len(templates)] for index in range(count))
        for count in counts
    }
    scales = {
        count: float(np.sqrt(TARGET_OCCUPIED_AREA_M2 / area_by_count[count])) for count in counts
    }
    material = {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0}
    scenes = []
    for count in counts:
        for layout in ("compact", "aligned", "dispersed"):
            objects = []
            for index, (definition, center) in enumerate(
                zip(
                    (templates[index % len(templates)] for index in range(count)),
                    _layout_centers(count, layout),
                )
            ):
                obj = _scaled_definition(
                    definition,
                    center,
                    scales[count],
                    material,
                )
                obj["object_id"] = f"c5_{count}_{index:02d}"
                obj["feature_size_m"] = definition["feature_size_m"] * scales[count]
                objects.append(obj)
            scene = {
                "schema_version": 3,
                "lineage_id": f"v3_c5_n{count}_{layout}",
                "family": "controlled_multiobject_layout",
                "stage": "C5",
                "split": "development",
                "object_count": count,
                "layout": layout,
                "matched_area_target_m2": TARGET_OCCUPIED_AREA_M2,
                "matched_count_group": f"c5_n{count}",
                "objects": objects,
                "frequency_hz": 1.0e9,
            }
            metrics = scene_metrics(scene)
            scene["scene_metrics"] = metrics
            scene["minimum_bbox_gap_m"] = metrics["minimum_axis_aligned_bbox_gap_m"]
            if count != len(objects) or scene["minimum_bbox_gap_m"] <= 0:
                raise ValueError(f"Invalid C5 geometry in {count}-object {layout} scene")
            scenes.append(scene)
    return scenes


def _object_area_from_definition(definition):
    from .curriculum_v2 import object_from_definition

    return _object_area(object_from_definition(definition))


def run_c5_layout_qualification(
    output,
    *,
    reference_levels=DEFAULT_REFERENCE_LEVELS,
    budgets=DEFAULT_BUDGETS,
    policies=DEFAULT_POLICIES,
    device="cuda:0",
    object_counts=(3, 5),
):
    """Qualify controlled 3/5-object development scenes and matched mesh policies."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(value) for value in reference_levels)
    budgets = tuple(int(value) for value in budgets)
    policies = tuple(policies)
    object_counts = tuple(int(value) for value in object_counts)
    if not object_counts or any(value not in (3, 5, 6, 8, 10) for value in object_counts):
        raise ValueError("C5 object counts must be selected from 3, 5, 6, 8, and 10")
    if len(levels) < 2 or tuple(sorted(set(levels))) != levels:
        raise ValueError("Reference levels must be at least two increasing budgets")
    if not budgets or any(
        value not in (32, 48, 64, 96, 128, 192, 256, 384, 512) for value in budgets
    ):
        raise ValueError("Unsupported C5 budget")
    if "uniform" not in policies or any(value not in CANDIDATE_NAMES for value in policies):
        raise ValueError("C5 policies must include uniform and use supported candidates")

    source_hashes = numerical_source_hashes()
    reference_engine_path = Path(__file__).with_name("c3_qualification_v2.py")
    campaign_source = Path(__file__).read_bytes()
    reference_engine_source = reference_engine_path.read_bytes()
    campaign_hash = hashlib.sha256(campaign_source + reference_engine_source).hexdigest()
    scene_rows = []
    for scene in generate_c5_controlled_scenes(object_counts):
        scene_root = output / "scenes" / scene["lineage_id"]
        qualification_path = scene_root / "qualification.json"
        qualification_path.parent.mkdir(parents=True, exist_ok=True)
        scene_hash = hashlib.sha256(
            json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        qualification, reference = None, None
        if qualification_path.exists():
            saved = json.loads(qualification_path.read_text())
            reference_path = qualification_path.with_suffix(".npz")
            if (
                saved.get("scene_sha256") == scene_hash
                and saved.get("source_hashes") == source_hashes
                and saved.get("campaign_source_sha256") == campaign_hash
                and saved.get("reference_levels") == list(levels)
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
            )
            if qualification.get("accepted"):
                reference_path = qualification_path.with_suffix(".npz")
                temporary = reference_path.with_name(reference_path.stem + ".tmp.npz")
                np.savez_compressed(temporary, complex_far_field=reference)
                os.replace(temporary, reference_path)
            _atomic_json(
                qualification_path,
                {
                    "scene_sha256": scene_hash,
                    "source_hashes": source_hashes,
                    "campaign_source_sha256": campaign_hash,
                    "reference_levels": list(levels),
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
                        material_samples=12,
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
                        }
                    )
        scoring = {}
        for cells in budgets:
            rows = [row for row in candidates if row["cells"] == cells]
            uniform = next((row for row in rows if row["policy"] == "uniform"), None)
            scores_by_exponent = {}
            if uniform and uniform["accepted"]:
                for exponent in (0.0, 0.02, 0.05, 0.1):
                    scores = {
                        row["policy"]: _score(row, uniform, exponent)
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
        threshold_summary = {}
        for target in (0.02, 0.05, 0.10):
            by_policy = {}
            for policy in policies:
                matching = [
                    row["cells"]
                    for row in candidates
                    if row["policy"] == policy
                    and row["accepted"]
                    and row["complex_relative_l2"] is not None
                    and row["width_relative_l2"] is not None
                    and row["complex_relative_l2"] <= target
                    and row["width_relative_l2"] <= target
                ]
                by_policy[policy] = min(matching) if matching else None
            threshold_summary[str(target)] = by_policy
        scene_rows.append(
            {
                "scene_id": scene["lineage_id"],
                "scene": scene,
                "object_count": scene["object_count"],
                "layout": scene["layout"],
                "occupied_area_m2": scene["scene_metrics"]["occupied_area_fraction"] * DOMAIN**2,
                "union_x_projection_fraction": scene["scene_metrics"][
                    "projected_x_support_fraction"
                ],
                "union_y_projection_fraction": scene["scene_metrics"][
                    "projected_y_support_fraction"
                ],
                "overall_extent_x_fraction": scene["scene_metrics"]["projected_x_extent_fraction"],
                "overall_extent_y_fraction": scene["scene_metrics"]["projected_y_extent_fraction"],
                "minimum_bbox_gap_m": scene["minimum_bbox_gap_m"],
                "qualification": qualification,
                "mesh_results": candidates,
                "scoring_sensitivity_by_cells": scoring,
                "minimum_cells_by_joint_error_target": threshold_summary,
            }
        )

    from collections import Counter

    statuses = Counter(row["status"] for scene in scene_rows for row in scene["mesh_results"])
    area_by_count = {
        str(count): [row["occupied_area_m2"] for row in scene_rows if row["object_count"] == count]
        for count in object_counts
    }
    report = {
        "schema_version": 1,
        "campaign": "c5_matched_area_layouts_v1"
        if object_counts == (3, 5)
        else "c5_matched_area_layouts_extended_v1",
        "object_counts": object_counts,
        "device": device,
        "reference_levels": levels,
        "budgets": budgets,
        "policies": policies,
        "frequencies_hz": FREQUENCIES,
        "monitor_bounds_m": MONITOR_BOUNDS,
        "matched_area_target_m2": TARGET_OCCUPIED_AREA_M2,
        "observed_area_by_object_count_m2": area_by_count,
        "source_hashes": source_hashes,
        "candidate_source_sha256": hashlib.sha256(
            Path(__file__).with_name("candidates_v2.py").read_bytes()
        ).hexdigest(),
        "reference_engine_source_sha256": hashlib.sha256(reference_engine_source).hexdigest(),
        "campaign_source_sha256": campaign_hash,
        "protocol": {
            "reference_complex_tolerance": SPATIAL_COMPLEX_TOLERANCE,
            "reference_width_tolerance": SPATIAL_WIDTH_TOLERANCE,
            "independent_probe_tolerance": PROBE_TOLERANCE,
            "pml_thickness_m": 0.12,
            "material_samples": 12,
            "durations_s": DURATIONS,
            "tail_tolerance": 1e-5,
            "dt_penalty_exponents": (0.0, 0.02, 0.05, 0.1),
        },
        "qualified_scene_count": sum(row["qualification"]["accepted"] for row in scene_rows),
        "scene_count": len(scene_rows),
        "candidate_case_count": sum(len(row["mesh_results"]) for row in scene_rows),
        "candidate_status_counts": dict(statuses),
        "scenes": scene_rows,
    }
    _atomic_json(output / "c5_layout_qualification_report.json", report)
    return report


def _score(row, uniform, exponent):
    return soft_dt_score(
        row["joint_scattering_loss"],
        row["dt"],
        uniform["dt"],
        exponent=exponent,
    )
