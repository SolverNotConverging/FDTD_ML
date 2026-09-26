"""Bounded CPU qualification sweep for dielectric C4 near-gap scenes."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .candidates_v2 import CANDIDATE_NAMES
from .curriculum_v2 import c4_gap_sampling_diagnostics, generate_c4_gap_sweep
from .scoring_v2 import relative_l2, soft_dt_score
from .simulation_v2 import FREQUENCIES, evaluate_case, numerical_source_hashes

DEFAULT_GAPS_M = (0.002, 0.005, 0.01, 0.02)
DEFAULT_REFERENCE_LEVELS = (48, 64, 96, 128, 192, 256, 384)
DEFAULT_BUDGETS = (32, 48, 64, 96, 128)
GAP_DURATIONS = (70e-9, 140e-9)
MAX_DURATION_PROBE_S = 140e-9
MONITOR_BOUNDS_M = (0.3, 0.9, 0.3, 0.9)
SPATIAL_COMPLEX_TOLERANCE = 0.005
SPATIAL_WIDTH_TOLERANCE = 0.01
PROBE_TOLERANCE = 0.01
NEAR_FIELD_TOLERANCE = 0.05


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_field(directory):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays["complex_far_field"].copy()


def _load_near_field(directory):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays["complex_near_field"].copy()


def _gap_sample_points(scene):
    """Five fixed normalized total-Ez samples along the vacuum gap centerline."""
    gap = float(scene["gap_m"])
    centers = np.asarray([obj["center_m"] for obj in scene["objects"]], dtype=float)
    center = centers.mean(axis=0)
    center[0] = float(scene["center_m"][0]) if "center_m" in scene else center[0]
    fractions = np.linspace(-0.4, 0.4, 5)
    points = np.column_stack((center[0] + fractions * gap, np.full(5, center[1])))
    return points


def _width(field):
    return 2 * np.pi * np.abs(field) ** 2


def _reference_probe(
    scene,
    cells,
    output,
    reference,
    reference_near_field,
    sample_points,
    *,
    name,
    device,
    base_duration,
):
    kwargs = {
        "durations": (2 * base_duration,),
        "pml_thickness": 0.12,
        "material_samples": 12,
        "monitor_bounds": MONITOR_BOUNDS_M,
    }
    if name == "duration":
        if 2 * base_duration > MAX_DURATION_PROBE_S:
            return {
                "status": "budget_limited",
                "accepted": False,
                "complex_change": None,
                "width_change": None,
                "near_field_change": None,
                "wall_seconds": 0.0,
                "Nt": None,
            }
        kwargs["durations"] = (2 * base_duration,)
    elif name == "quadrature":
        kwargs["durations"] = (base_duration,)
        kwargs["material_samples"] = 24
    elif name == "contour":
        kwargs["durations"] = (base_duration,)
        kwargs["monitor_bounds"] = (0.32, 0.88, 0.32, 0.88)
    elif name == "pml":
        kwargs["durations"] = (base_duration,)
        kwargs["pml_thickness"] = 0.15
    else:
        raise ValueError(f"Unknown reference probe: {name}")
    directory = Path(output) / "probes" / f"{scene['lineage_id']}_n{cells}_{name}"
    record, _ = evaluate_case(
        scene,
        cells,
        0.0,
        "uniform",
        directory,
        device=device,
        reference=reference,
        reference_near_field=reference_near_field,
        field_sample_points=sample_points,
        **kwargs,
    )
    field = _load_field(directory) if record.get("accepted") else None
    near_field = _load_near_field(directory) if record.get("accepted") else None
    return {
        "status": record["status"],
        "accepted": bool(record.get("accepted")),
        "complex_change": relative_l2(field, reference) if field is not None else None,
        "width_change": relative_l2(_width(field), _width(reference))
        if field is not None
        else None,
        "near_field_change": relative_l2(near_field, reference_near_field)
        if near_field is not None
        else None,
        "wall_seconds": record.get("wall_seconds"),
        "Nt": record.get("Nt"),
    }


def _qualify_gap_reference(scene, output, *, reference_levels, device):
    observations = []
    previous = None
    previous_near_field = None
    selected_cells, reference, reference_near_field, selected_record = None, None, None, None
    sample_points = _gap_sample_points(scene)
    for cells in reference_levels:
        directory = Path(output) / "references" / f"{scene['lineage_id']}_n{cells}"
        record, _ = evaluate_case(
            scene,
            cells,
            0.0,
            "uniform",
            directory,
            device=device,
            pml_thickness=0.12,
            material_samples=12,
            durations=GAP_DURATIONS,
            tail_tolerance=1e-5,
            monitor_bounds=MONITOR_BOUNDS_M,
            field_sample_points=sample_points,
        )
        row = {
            "cells": int(cells),
            "status": record["status"],
            "accepted": bool(record.get("accepted")),
            "wall_seconds": record.get("wall_seconds"),
            "Nt": record.get("Nt"),
            "dt": record.get("dt"),
            "tail_peak_over_global_peak": record.get("tail_peak_over_global_peak"),
        }
        observations.append(row)
        if not record.get("accepted"):
            break
        field = _load_field(directory)
        near_field = _load_near_field(directory)
        if previous is not None:
            complex_change = relative_l2(field, previous)
            width_change = relative_l2(_width(field), _width(previous))
            row["complex_change_from_previous"] = complex_change
            row["width_change_from_previous"] = width_change
            near_field_change = relative_l2(near_field, previous_near_field)
            row["near_field_change_from_previous"] = near_field_change
            if (
                complex_change <= SPATIAL_COMPLEX_TOLERANCE
                and width_change <= SPATIAL_WIDTH_TOLERANCE
                and near_field_change <= NEAR_FIELD_TOLERANCE
            ):
                selected_cells, reference, reference_near_field, selected_record = (
                    cells,
                    field,
                    near_field,
                    record,
                )
                break
        previous = field
        previous_near_field = near_field

    if selected_cells is None:
        last_status = observations[-1]["status"] if observations else "no_reference_levels"
        status = last_status if last_status != "accepted" else "reference_not_converged"
        return (
            {
                "accepted": False,
                "status": status,
                "selected_cells": None,
                "observations": observations,
                "independent_probes": {},
                "reference_uncertainty_relative": None,
            },
            None,
            None,
        )

    probe_rows = {
        name: _reference_probe(
            scene,
            selected_cells,
            output,
            reference,
            reference_near_field,
            sample_points,
            name=name,
            device=device,
            base_duration=selected_record["attempts"][-1]["duration_s"],
        )
        for name in ("duration", "quadrature", "contour", "pml")
    }
    spatial_uncertainty = max(
        observations[-1]["complex_change_from_previous"],
        observations[-1]["width_change_from_previous"],
        observations[-1]["near_field_change_from_previous"],
    )
    probe_changes = [
        value
        for row in probe_rows.values()
        for value in (row["complex_change"], row["width_change"], row["near_field_change"])
        if value is not None
    ]
    probes_pass = all(
        row["status"] == "accepted"
        and row["complex_change"] is not None
        and row["width_change"] is not None
        and row["near_field_change"] is not None
        and row["complex_change"] <= PROBE_TOLERANCE
        and row["width_change"] <= PROBE_TOLERANCE
        and row["near_field_change"] <= NEAR_FIELD_TOLERANCE
        for row in probe_rows.values()
    )
    uncertainty = (
        max([spatial_uncertainty, *probe_changes]) if probe_changes else spatial_uncertainty
    )
    status = "accepted" if probes_pass else "probe_sensitive"
    report = {
        "accepted": probes_pass,
        "status": status,
        "selected_cells": int(selected_cells),
        "observations": observations,
        "independent_probes": probe_rows,
        "reference_uncertainty_relative": float(uncertainty),
        "dt": selected_record.get("dt"),
        "near_field_sample_points_m": sample_points.tolist(),
        "near_field_tolerance": NEAR_FIELD_TOLERANCE,
    }
    return report, reference, reference_near_field


def run_c4_gap_sweep(
    output,
    *,
    gaps_m=DEFAULT_GAPS_M,
    reference_levels=DEFAULT_REFERENCE_LEVELS,
    budgets=DEFAULT_BUDGETS,
    policies=("uniform", "center", "interface", "wide", "hybrid"),
    device="cuda:0",
):
    """Run a resumable dielectric gap sweep and store convergence/mesh evidence.

    The sweep uses one incidence angle at 0.8/1.0/1.2 GHz. A gap is reference
    qualified only when successive uniform grids converge in both far-field
    scattering and five fixed gap-line near-field samples, and duration, PML,
    contour, and quadrature probes stay within their declared tolerances. Cases
    that fail either gate remain in the output with their failure status.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(value) for value in reference_levels)
    requested_budgets = tuple(int(value) for value in budgets)
    requested_policies = tuple(policies)
    if (
        len(levels) < 2
        or tuple(sorted(set(levels))) != levels
        or any(value < 8 for value in levels)
    ):
        raise ValueError("Reference levels must be at least two strictly increasing cell counts")
    if not requested_budgets or any(value < 8 for value in requested_budgets):
        raise ValueError("At least one positive spatial budget is required")
    if (
        not requested_policies
        or len(set(requested_policies)) != len(requested_policies)
        or any(policy not in CANDIDATE_NAMES for policy in requested_policies)
        or "uniform" not in requested_policies
    ):
        raise ValueError("Policies must be unique supported candidates and include uniform")
    scenes = generate_c4_gap_sweep(gaps_m)
    report_scenes = []
    source_hashes = numerical_source_hashes()
    for scene in scenes:
        qualification_path = output / "qualifications" / f"{scene['lineage_id']}.json"
        qualification_path.parent.mkdir(parents=True, exist_ok=True)
        if qualification_path.exists():
            previous_report = json.loads(qualification_path.read_text())
            if (
                previous_report.get("scene_sha256")
                == hashlib.sha256(
                    json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                and previous_report.get("source_hashes") == source_hashes
                and previous_report.get("reference_levels") == list(levels)
                and (
                    not previous_report["qualification"].get("accepted")
                    or qualification_path.with_suffix(".npz").exists()
                )
            ):
                qualification = previous_report["qualification"]
                reference = None
                reference_near_field = None
                if qualification.get("accepted"):
                    with np.load(qualification_path.with_suffix(".npz")) as arrays:
                        reference = arrays["complex_far_field"].copy()
                        reference_near_field = arrays["complex_near_field"].copy()
            else:
                qualification, reference, reference_near_field = _qualify_gap_reference(
                    scene, output, reference_levels=levels, device=device
                )
        else:
            qualification, reference, reference_near_field = _qualify_gap_reference(
                scene, output, reference_levels=levels, device=device
            )

        if qualification.get("accepted"):
            reference_path = qualification_path.with_suffix(".npz")
            if reference is not None:
                temporary = reference_path.with_name(reference_path.stem + ".tmp.npz")
                np.savez_compressed(
                    temporary,
                    complex_far_field=reference,
                    complex_near_field=reference_near_field,
                )
                os.replace(temporary, reference_path)
            elif not reference_path.exists():
                qualification["accepted"] = False
                qualification["status"] = "missing_reference_artifact"

        mesh_rows = []
        for cells in requested_budgets:
            for policy_name in requested_policies:
                directory = output / "meshes" / f"{scene['lineage_id']}_n{cells}_{policy_name}"
                record, _ = evaluate_case(
                    scene,
                    cells,
                    0.0,
                    policy_name,
                    directory,
                    device=device,
                    reference=reference if qualification.get("accepted") else None,
                    reference_near_field=(
                        reference_near_field if qualification.get("accepted") else None
                    ),
                    field_sample_points=_gap_sample_points(scene),
                    pml_thickness=0.12,
                    material_samples=12,
                    durations=GAP_DURATIONS,
                    tail_tolerance=1e-5,
                    monitor_bounds=MONITOR_BOUNDS_M,
                )
                mesh_rows.append(
                    {
                        "cells": cells,
                        "method": policy_name,
                        "status": record["status"],
                        "accepted": bool(record.get("accepted")),
                        "complex_relative_l2": record.get("complex_relative_l2"),
                        "width_relative_l2": record.get("width_relative_l2"),
                        "near_field_relative_l2": record.get("near_field_relative_l2"),
                        "joint_scattering_loss": record.get("joint_scattering_loss"),
                        "dt": record.get("dt"),
                        "Nt": record.get("Nt"),
                        "wall_seconds": record.get("wall_seconds"),
                        "uniform_repair_fraction": record.get("uniform_repair_fraction"),
                    }
                )

        sampling_by_budget = {}
        for cells in requested_budgets:
            diagnostics = c4_gap_sampling_diagnostics(scene, cells)
            diagnostics["qualification_status"] = (
                "reference_qualified" if qualification.get("accepted") else "reference_unqualified"
            )
            diagnostics["mesh_status_by_policy"] = {
                mesh["method"]: mesh["status"] for mesh in mesh_rows if mesh["cells"] == cells
            }
            sampling_by_budget[str(cells)] = diagnostics

        scoring_sensitivity = {}
        for cells in requested_budgets:
            rows = [row for row in mesh_rows if row["cells"] == cells]
            uniform = next(row for row in rows if row["method"] == "uniform")
            if not uniform["accepted"]:
                scoring_sensitivity[str(cells)] = {
                    "status": "uniform_unsettled",
                    "scores_by_dt_exponent": {},
                }
                continue
            by_exponent = {}
            for exponent in (0.0, 0.02, 0.05, 0.1):
                scores = {
                    row["method"]: soft_dt_score(
                        row["joint_scattering_loss"],
                        row["dt"],
                        uniform["dt"],
                        exponent=exponent,
                    )
                    for row in rows
                    if row["accepted"] and row["joint_scattering_loss"] is not None
                }
                by_exponent[str(exponent)] = {
                    "scores": scores,
                    "best_candidate": min(scores, key=scores.get) if scores else None,
                }
            scoring_sensitivity[str(cells)] = {
                "status": "scored",
                "scores_by_dt_exponent": by_exponent,
            }

        row = {
            "scene_id": scene["lineage_id"],
            "gap_m": scene["gap_m"],
            "gap_wavelengths": scene["gap_wavelengths"],
            "radius_m": scene["objects"][0]["radius_m"],
            "object_count": len(scene["objects"]),
            "material": scene["objects"][0]["material"],
            "source_frequency_hz": scene["frequency_hz"],
            "observation_frequencies_hz": list(FREQUENCIES),
            "incidence_angle_rad": 0.0,
            "qualification": qualification,
            "sampling_by_budget": sampling_by_budget,
            "mesh_results": mesh_rows,
            "scoring_sensitivity_by_cells": scoring_sensitivity,
        }
        report_scenes.append(row)
        scene_sha = hashlib.sha256(
            json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _atomic_json(
            qualification_path,
            {
                "scene_sha256": scene_sha,
                "source_hashes": source_hashes,
                "reference_levels": levels,
                "qualification": qualification,
            },
        )

    result = {
        "schema_version": 2,
        "protocol": "c4_dielectric_gap_sweep_v2_near_field",
        "device": device,
        "gaps_m": [scene["gap_m"] for scene in scenes],
        "reference_levels": levels,
        "budgets": requested_budgets,
        "policies": requested_policies,
        "monitor_bounds_m": MONITOR_BOUNDS_M,
        "near_field_definition": "five total-Ez DFT samples divided by the local incident-Ez DFT, equally spaced along the vacuum gap centerline",
        "spatial_complex_tolerance": SPATIAL_COMPLEX_TOLERANCE,
        "spatial_width_tolerance": SPATIAL_WIDTH_TOLERANCE,
        "probe_tolerance": PROBE_TOLERANCE,
        "near_field_tolerance": NEAR_FIELD_TOLERANCE,
        "duration_limit_s": MAX_DURATION_PROBE_S,
        "duration_attempts_s": GAP_DURATIONS,
        "source_hashes": source_hashes,
        "scenes": report_scenes,
        "summary": {
            "qualified_gap_count": sum(
                row["qualification"].get("accepted", False) for row in report_scenes
            ),
            "unqualified_gap_count": sum(
                not row["qualification"].get("accepted", False) for row in report_scenes
            ),
            "candidate_cases": sum(len(row["mesh_results"]) for row in report_scenes),
            "candidate_status_counts": dict(
                sorted(
                    Counter(
                        mesh["status"] for scene in report_scenes for mesh in scene["mesh_results"]
                    ).items()
                )
            ),
            "reference_status_counts": dict(
                sorted(Counter(scene["qualification"]["status"] for scene in report_scenes).items())
            ),
        },
    }
    _atomic_json(output / "c4_gap_sweep_report.json", result)
    return result
