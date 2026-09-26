"""Bounded physical qualification for ordinary separated two-object scenes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import scene_metrics
from .scoring_v2 import relative_l2, soft_dt_score
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes

DEFAULT_REFERENCE_LEVELS = (64, 96, 128, 192, 256, 384)
DEFAULT_BUDGETS = (32, 48, 64, 96, 128)
DEFAULT_POLICIES = ("uniform", "interface", "center", "hybrid")
DURATIONS = (70e-9, 140e-9)
MONITOR_BOUNDS = (0.30, 0.90, 0.30, 0.90)
SPATIAL_COMPLEX_TOLERANCE = 0.005
SPATIAL_WIDTH_TOLERANCE = 0.01
PROBE_TOLERANCE = 0.01


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_field(directory):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays["complex_far_field"].copy()


def _width(field):
    return 2 * np.pi * np.abs(field) ** 2


def _definition(object_id, center, radius, *, epsilon_r=4.0, angle=0.0):
    return {
        "object_id": object_id,
        "shape": "ellipse" if angle else "circle",
        "center_m": [float(value) for value in center],
        **({"radius_m": float(radius[0])} if len(radius) == 1 else {}),
        **(
            {"radii_m": [float(value) for value in radius], "angle_rad": float(angle)}
            if len(radius) == 2
            else {}
        ),
        "feature_size_m": float(min(radius)),
        "material": {"kind": "dielectric", "epsilon_r": float(epsilon_r), "sigma_e_s_per_m": 0.0},
    }


def generate_c3_pair_sweep():
    """Four controlled pair cases covering separation, orientation, and size ratio."""
    cases = (
        ("separation_near", (0.045, 0.045), (0.0, 0.0), 0.20, 0.0),
        ("separation_far", (0.045, 0.045), (0.0, 0.0), 0.34, 0.0),
        ("size_ratio_2", (0.040, 0.080), (0.0, 0.0), 0.26, 0.0),
        ("diagonal_pair", (0.055, 0.055), (0.0, 0.0), 0.21, np.pi / 4),
    )
    rows = []
    center = np.asarray((0.60, 0.60))
    for index, (name, radii, _, separation, angle) in enumerate(cases):
        direction = np.asarray((np.cos(angle), np.sin(angle)))
        positions = (center - separation * direction / 2, center + separation * direction / 2)
        objects = [
            _definition(f"pair_{index}_{side}", position, (radius,))
            for side, position, radius in zip(("a", "b"), positions, radii)
        ]
        if name == "size_ratio_2":
            # Keep the pair's bounding-box gap positive after changing radii.
            objects[0]["radius_m"] = 0.04
            objects[1]["radius_m"] = 0.08
            objects[0]["feature_size_m"] = 0.04
            objects[1]["feature_size_m"] = 0.08
        scene = {
            "schema_version": 3,
            "lineage_id": f"v3_c3_{index:02d}_{name}",
            "family": "controlled_two_object_pair",
            "stage": "C3",
            "split": "development",
            "objects": objects,
            "pair_separation_m": float(separation),
            "pair_orientation_rad": float(angle),
            "pair_size_ratio": float(max(radii) / min(radii)),
            "control_name": name,
            "frequency_hz": 1.0e9,
        }
        metrics = scene_metrics(scene)
        scene["minimum_bbox_gap_m"] = metrics["minimum_axis_aligned_bbox_gap_m"]
        if scene["minimum_bbox_gap_m"] <= 0:
            raise ValueError(f"C3 scene {name} does not have a positive bounding-box gap")
        rows.append(scene)
    return rows


def _qualify_reference(
    scene,
    root,
    *,
    levels,
    device,
    monitor_bounds=MONITOR_BOUNDS,
    durations=DURATIONS,
):
    observations = []
    previous_field = None
    selected = None
    for cells in levels:
        directory = root / "references" / f"uniform_n{cells}"
        record, _ = evaluate_case(
            scene,
            cells,
            0.0,
            "uniform",
            directory,
            device=device,
            pml_thickness=0.12,
            material_samples=12,
            durations=durations,
            tail_tolerance=1e-5,
            monitor_bounds=monitor_bounds,
        )
        observation = {
            "cells": cells,
            "status": record["status"],
            "accepted": bool(record.get("accepted")),
            "dt": record.get("dt"),
            "Nt": record.get("Nt"),
            "wall_seconds": record.get("wall_seconds"),
        }
        observations.append(observation)
        if not record.get("accepted"):
            break
        field = _load_field(directory)
        if previous_field is not None:
            complex_change = relative_l2(field, previous_field)
            width_change = relative_l2(_width(field), _width(previous_field))
            observation.update(
                {
                    "complex_change_from_previous": complex_change,
                    "width_change_from_previous": width_change,
                }
            )
            if (
                complex_change <= SPATIAL_COMPLEX_TOLERANCE
                and width_change <= SPATIAL_WIDTH_TOLERANCE
            ):
                selected = (cells, field, record)
                break
        previous_field = field

    if selected is None:
        return {
            "accepted": False,
            "status": "reference_not_converged",
            "observations": observations,
            "independent_probes": {},
            "reference_uncertainty_relative": None,
        }, None

    cells, reference, record = selected
    base_duration = record["attempts"][-1]["duration_s"]
    probes = {}
    a, b, c, d = monitor_bounds
    contour_margin = min(0.02, 0.25 * min(b - a, d - c))
    specs = {
        "duration": {"durations": (2 * base_duration,)},
        "quadrature": {"durations": (base_duration,), "material_samples": 24},
        "contour": {
            "durations": (base_duration,),
            "monitor_bounds": (
                a + contour_margin,
                b - contour_margin,
                c + contour_margin,
                d - contour_margin,
            ),
        },
        "pml": {"durations": (base_duration,), "pml_thickness": 0.15},
    }
    for name, changes in specs.items():
        directory = root / "probes" / f"n{cells}_{name}"
        probe_kwargs = {
            "device": device,
            "reference": reference,
            "pml_thickness": 0.12,
            "material_samples": 12,
            "tail_tolerance": 1e-5,
            "monitor_bounds": monitor_bounds,
            **changes,
        }
        probe, _ = evaluate_case(
            scene,
            cells,
            0.0,
            "uniform",
            directory,
            **probe_kwargs,
        )
        field = _load_field(directory) if probe.get("accepted") else None
        probes[name] = {
            "status": probe["status"],
            "accepted": bool(probe.get("accepted")),
            "complex_change": relative_l2(field, reference) if field is not None else None,
            "width_change": relative_l2(_width(field), _width(reference))
            if field is not None
            else None,
            "wall_seconds": probe.get("wall_seconds"),
        }
    probes_pass = all(
        row["accepted"]
        and row["complex_change"] is not None
        and row["width_change"] is not None
        and row["complex_change"] <= PROBE_TOLERANCE
        and row["width_change"] <= PROBE_TOLERANCE
        for row in probes.values()
    )
    spatial_uncertainty = max(
        observations[-1]["complex_change_from_previous"],
        observations[-1]["width_change_from_previous"],
    )
    probe_changes = [
        value
        for row in probes.values()
        for value in (row["complex_change"], row["width_change"])
        if value is not None
    ]
    uncertainty = (
        max([spatial_uncertainty, *probe_changes]) if probe_changes else spatial_uncertainty
    )
    return (
        {
            "accepted": probes_pass,
            "status": "accepted" if probes_pass else "probe_sensitive",
            "selected_cells": int(cells),
            "observations": observations,
            "independent_probes": probes,
            "reference_uncertainty_relative": float(uncertainty),
            "reference_complex_tolerance": SPATIAL_COMPLEX_TOLERANCE,
            "reference_width_tolerance": SPATIAL_WIDTH_TOLERANCE,
            "probe_tolerance": PROBE_TOLERANCE,
            "dt": record.get("dt"),
        },
        reference,
    )


def run_c3_pair_qualification(
    output,
    *,
    reference_levels=DEFAULT_REFERENCE_LEVELS,
    budgets=DEFAULT_BUDGETS,
    policies=DEFAULT_POLICIES,
    device="cuda:0",
):
    """Run a resumable four-scene C3 physical and fixed-policy study."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(value) for value in reference_levels)
    budgets = tuple(int(value) for value in budgets)
    policies = tuple(policies)
    if len(levels) < 2 or tuple(sorted(set(levels))) != levels:
        raise ValueError("Reference levels must be at least two increasing budgets")
    if not budgets or any(
        value not in (32, 48, 64, 96, 128, 192, 256, 384, 512) for value in budgets
    ):
        raise ValueError("Unsupported C3 budget")
    if "uniform" not in policies or any(value not in CANDIDATE_NAMES for value in policies):
        raise ValueError("C3 policies must include uniform and use supported candidates")

    source_hashes = numerical_source_hashes()
    scene_rows = []
    for scene in generate_c3_pair_sweep():
        scene_root = output / "scenes" / scene["lineage_id"]
        qualification_path = scene_root / "qualification.json"
        scene_hash = hashlib.sha256(
            json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        qualification = None
        reference = None
        if qualification_path.exists():
            saved = json.loads(qualification_path.read_text())
            reference_path = qualification_path.with_suffix(".npz")
            if (
                saved.get("scene_sha256") == scene_hash
                and saved.get("source_hashes") == source_hashes
                and saved.get("campaign_source_sha256")
                == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                and saved.get("reference_levels") == list(levels)
                and (not saved["qualification"].get("accepted") or reference_path.exists())
            ):
                qualification = saved["qualification"]
                if qualification.get("accepted"):
                    with np.load(reference_path) as arrays:
                        reference = arrays["complex_far_field"].copy()
        if qualification is None:
            qualification, reference = _qualify_reference(
                scene, scene_root, levels=levels, device=device
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
                    "campaign_source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "reference_levels": list(levels),
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

        sensitivity = {}
        for cells in budgets:
            matched = [row for row in candidates if row["cells"] == cells]
            uniform = next((row for row in matched if row["policy"] == "uniform"), None)
            exponents = {}
            if uniform and uniform["accepted"]:
                for exponent in (0.0, 0.02, 0.05, 0.1):
                    scores = {
                        row["policy"]: soft_dt_score(
                            row["joint_scattering_loss"],
                            row["dt"],
                            uniform["dt"],
                            exponent=exponent,
                        )
                        for row in matched
                        if row["accepted"] and row["joint_scattering_loss"] is not None
                    }
                    exponents[str(exponent)] = {
                        "scores": scores,
                        "best_policy": min(scores, key=scores.get) if scores else None,
                    }
            sensitivity[str(cells)] = {
                "status": "scored" if exponents else "uniform_unsettled_or_unqualified",
                "scores_by_dt_exponent": exponents,
            }

        scene_rows.append(
            {
                "scene_id": scene["lineage_id"],
                "scene": scene,
                "control_name": scene["control_name"],
                "pair_separation_m": scene["pair_separation_m"],
                "pair_orientation_rad": scene["pair_orientation_rad"],
                "pair_size_ratio": scene["pair_size_ratio"],
                "minimum_bbox_gap_m": scene["minimum_bbox_gap_m"],
                "qualification": qualification,
                "mesh_results": candidates,
                "scoring_sensitivity_by_cells": sensitivity,
            }
        )

    from collections import Counter

    statuses = Counter(row["status"] for scene in scene_rows for row in scene["mesh_results"])
    report = {
        "schema_version": 1,
        "campaign": "c3_controlled_pairs_v1",
        "device": device,
        "reference_levels": levels,
        "budgets": budgets,
        "policies": policies,
        "frequencies_hz": FREQUENCIES,
        "monitor_bounds_m": MONITOR_BOUNDS,
        "source_hashes": source_hashes,
        "campaign_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    _atomic_json(output / "c3_pair_qualification_report.json", report)
    return report
