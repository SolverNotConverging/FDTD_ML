"""Independent reference checks for supported continuous-geometry scenes."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .candidates_v2 import candidate_axes
from .curriculum_v2 import DOMAIN, objects_from_scene
from .geometry import Circle
from .grid import Grid
from .monitor_v2 import widest_non_pml_monitor_bounds
from .scoring_v2 import relative_l2
from .simulation_v2 import DURATIONS, evaluate_case, numerical_source_hashes


def _load_field(directory):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays["complex_far_field"].copy()


def _load_compact_field(directory):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays["complex_far_field"].copy(), arrays.get("reference_complex_far_field")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def qualify_compact_reference(
    scene,
    angle_index,
    angle,
    output,
    policy,
    *,
    calibration_profile_sha256,
    device="cuda:0",
    deadline=None,
):
    """Use the frozen compact-protocol convergence rule for one scene and angle."""
    output = Path(output)
    report_path = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    policy_sha256 = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_hashes = numerical_source_hashes()
    qualification_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if (
            previous.get("protocol") == "compact_adaptive_v1"
            and previous.get("policy_sha256") == policy_sha256
            and previous.get("calibration_profile_sha256") == calibration_profile_sha256
            and previous.get("source_hashes") == source_hashes
            and previous.get("qualification_sha256") == qualification_sha256
            and previous.get("terminal")
            and (not previous.get("accepted") or report_path.with_suffix(".npz").exists())
        ):
            return previous

    levels = tuple(int(value) for value in policy["spatial_levels"])
    if len(levels) < 2 or tuple(sorted(set(levels))) != levels:
        raise ValueError("Compact reference levels must be strictly increasing")
    complex_tolerance = float(policy["complex_tolerance"])
    width_tolerance = float(policy["width_tolerance"])
    pml_thickness = float(policy["pml_thickness_m"])
    material_samples = int(policy["material_samples"])
    durations = tuple(float(value) for value in policy["durations_s"])
    objects = objects_from_scene(scene)
    observations = []
    previous_field = None
    selected_cells = None
    accepted_field = None
    accepted_record = None
    reason = None
    incomplete = False
    for cells in levels:
        if deadline is not None and time.time() >= deadline:
            reason, incomplete = "deadline", True
            break
        directory = output / "compact_references" / f"{scene['lineage_id']}_a{angle_index}_n{cells}"
        # Larger reference grids use explicit uniform axes. The candidate policy
        # only enumerates pilot mesh budgets up to 512 cells per axis.
        explicit_uniform = (
            (np.linspace(0.0, DOMAIN, cells + 1), np.linspace(0.0, DOMAIN, cells + 1))
            if cells > 512
            else None
        )
        record, _ = evaluate_case(
            scene,
            cells,
            angle,
            "learned" if explicit_uniform is not None else "uniform",
            directory,
            device=device,
            selected_axes=explicit_uniform,
            durations=durations,
            tail_tolerance=float(policy["reference_tail_ratio"]),
            pml_thickness=pml_thickness,
            material_samples=material_samples,
            source_hashes=source_hashes,
        )
        row = {
            "cells": cells,
            "status": record["status"],
            "accepted": bool(record.get("accepted")),
            "wall_seconds": record.get("wall_seconds"),
            "Nt": record.get("Nt"),
            "dt": record.get("dt"),
            "tail": record.get("tail_peak_over_global_peak"),
            "monitor_bounds_m": record.get("monitor_bounds"),
        }
        observations.append(row)
        if not record.get("accepted"):
            reason = f"n{cells}_{record['status']}"
            break
        field, analytic_reference = _load_compact_field(directory)
        if analytic_reference is not None:
            row["analytic_complex_error"] = relative_l2(field, analytic_reference)
            row["analytic_width_error"] = relative_l2(
                2 * np.pi * np.abs(field) ** 2,
                2 * np.pi * np.abs(analytic_reference) ** 2,
            )
        if previous_field is not None:
            complex_change = relative_l2(field, previous_field)
            width_change = relative_l2(
                2 * np.pi * np.abs(field) ** 2, 2 * np.pi * np.abs(previous_field) ** 2
            )
            row["complex_change_from_previous"] = complex_change
            row["width_change_from_previous"] = width_change
            if complex_change <= complex_tolerance and width_change <= width_tolerance:
                selected_cells = cells
                accepted_field = analytic_reference if analytic_reference is not None else field
                accepted_record = record
                break
        previous_field = field

    accepted = selected_cells is not None
    statuses = {row["status"] for row in observations if row["status"] != "accepted"}
    status = (
        "accepted"
        if accepted
        else (
            "incomplete"
            if incomplete
            else "geometry_incompatible"
            if "geometry_incompatible" in statuses
            else "budget_limited"
            if "budget_limited" in statuses
            else "unsettled"
            if "unsettled" in statuses
            else "reference_not_converged"
        )
    )
    uncertainty = None
    if accepted:
        uncertainty = max(
            observations[-1]["complex_change_from_previous"],
            observations[-1]["width_change_from_previous"],
            observations[-1].get("analytic_complex_error", 0.0),
            observations[-1].get("analytic_width_error", 0.0),
        )
    report = {
        "schema_version": 3,
        "protocol": "compact_adaptive_v1",
        "terminal": not incomplete,
        "accepted": accepted,
        "status": status,
        "reason": reason,
        "scene_id": scene["lineage_id"],
        "angle_index": angle_index,
        "selected_cells": selected_cells,
        "observations": observations,
        "source_hashes": source_hashes,
        "monitor_policy": "per_grid_widest_non_pml_v1",
        "qualification_sha256": qualification_sha256,
        "policy_sha256": policy_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
        "reference_source": "analytic_circle"
        if accepted and len(objects) == 1 and isinstance(objects[0], Circle)
        else "converged_uniform_fdtd"
        if accepted
        else None,
        "dt": accepted_record["dt"] if accepted else None,
        "reference_uncertainty_relative": uncertainty,
    }
    if accepted:
        field_path = report_path.with_suffix(".npz")
        temporary = field_path.with_name(field_path.stem + ".tmp.npz")
        np.savez_compressed(temporary, complex_far_field=accepted_field)
        os.replace(temporary, field_path)
    _write_json(report_path, report)
    return report


def qualify_reference(scene, angle_index, angle, output, *, device="cuda:0", deadline=None):
    """Escalate spatial resolution and check separate observation dimensions."""
    output = Path(output)
    report_path = output / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = numerical_source_hashes()
    qualification_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if (
            previous.get("source_hashes") == source_hashes
            and previous.get("qualification_sha256") == qualification_sha256
            and previous.get("terminal")
            and (not previous.get("accepted") or report_path.with_suffix(".npz").exists())
        ):
            return previous
    observations = {}
    probe_records = {}
    accepted_field = None
    selected_cells = None
    reason = None
    base_record = None
    incomplete = False
    for cells in (192, 256, 384, 512):
        if deadline is not None and time.time() >= deadline:
            reason, incomplete = "deadline", True
            break
        reference_directory = (
            output / "references" / f"{scene['lineage_id']}_a{angle_index}_n{cells}"
        )
        axis, _, _ = candidate_axes(scene, cells, "uniform")
        objects = objects_from_scene(scene)
        monitor_bounds = widest_non_pml_monitor_bounds(
            Grid(axis, axis, max_ratio=3), [obj.bounds for obj in objects], 0.15
        )
        record, _ = evaluate_case(
            scene,
            cells,
            angle,
            "uniform",
            reference_directory,
            device=device,
            source_hashes=source_hashes,
            monitor_bounds=monitor_bounds,
        )
        observations[str(cells)] = {
            "status": record["status"],
            "accepted": record.get("accepted"),
            "tail": record.get("tail_peak_over_global_peak"),
            "wall_seconds": record.get("wall_seconds"),
        }
        if not record.get("accepted"):
            reason = f"n{cells}_{record['status']}"
            break
        if cells < 384:
            continue
        field = _load_field(reference_directory)
        prior = _load_field(
            output
            / "references"
            / f"{scene['lineage_id']}_a{angle_index}_n{256 if cells == 384 else 384}"
        )
        spatial = relative_l2(field, prior)
        observations[str(cells)]["complex_spatial_change"] = spatial
        if spatial > 0.005:
            reason = f"n{cells}_spatial_change"
            continue
        reference_field = field
        if len(objects) == 1 and isinstance(objects[0], Circle):
            with np.load(reference_directory / "spectra.npz") as arrays:
                reference_field = arrays["reference_complex_far_field"].copy()
            analytic_change = relative_l2(field, reference_field)
            observations[str(cells)]["analytic_complex_error"] = analytic_change
            if analytic_change > 0.01:
                reason = f"n{cells}_analytic_error"
                continue
        base_duration = record["attempts"][-1]["duration_s"]
        next_duration = next((value for value in DURATIONS if value > base_duration), 700e-9)
        base_monitor = record["monitor_bounds"]
        a, b, c, d = base_monitor
        probe_spec = {
            "duration": {
                "durations": (next_duration,),
                "material_samples": 12,
                "monitor_bounds": base_monitor,
                "pml_thickness": 0.12,
            },
            "quadrature": {
                "durations": (base_duration,),
                "material_samples": 24,
                "monitor_bounds": base_monitor,
                "pml_thickness": 0.12,
            },
            "contour": {
                "durations": (base_duration,),
                "material_samples": 12,
                "monitor_bounds": (a + 0.02, b - 0.02, c + 0.02, d - 0.02),
                "pml_thickness": 0.12,
            },
            "pml": {
                "durations": (base_duration,),
                "material_samples": 12,
                "monitor_bounds": base_monitor,
                "pml_thickness": 0.15,
            },
        }
        changes = {}
        failed = False
        for name, kwargs in probe_spec.items():
            if deadline is not None and time.time() >= deadline:
                reason, incomplete, failed = "deadline", True, True
                break
            probe_dir = (
                output
                / "reference_probes"
                / f"{scene['lineage_id']}_a{angle_index}_n{cells}_{name}"
            )
            probe, _ = evaluate_case(
                scene,
                cells,
                angle,
                "uniform",
                probe_dir,
                device=device,
                reference=field,
                source_hashes=source_hashes,
                **kwargs,
            )
            change = relative_l2(_load_field(probe_dir), field) if probe.get("accepted") else None
            changes[name] = {"status": probe["status"], "complex_change": change}
            if change is None or change > 0.005:
                failed = True
        probe_records[str(cells)] = changes
        if incomplete:
            break
        if failed:
            reason = f"n{cells}_independent_probe"
            continue
        selected_cells, accepted_field, base_record = cells, reference_field, record
        reason = None
        break
    accepted = selected_cells is not None
    failing_statuses = {
        item["status"] for item in observations.values() if item["status"] != "accepted"
    }
    if accepted:
        status = "accepted"
    elif incomplete:
        status = "incomplete"
    elif "geometry_incompatible" in failing_statuses:
        status = "geometry_incompatible"
    elif "budget_limited" in failing_statuses:
        status = "budget_limited"
    elif "unsettled" in failing_statuses:
        status = "unsettled"
    else:
        status = "reference_not_converged"
    uncertainty = None
    if accepted:
        uncertainty = max(
            observations[str(selected_cells)]["complex_spatial_change"],
            *(probe["complex_change"] for probe in probe_records[str(selected_cells)].values()),
        )
    report = {
        "schema_version": 2,
        "terminal": not incomplete,
        "accepted": accepted,
        "status": status,
        "reason": reason,
        "scene_id": scene["lineage_id"],
        "angle_index": angle_index,
        "selected_cells": selected_cells,
        "observations": observations,
        "independent_probes": probe_records,
        "source_hashes": source_hashes,
        "qualification_sha256": qualification_sha256,
        "reference_source": (
            "analytic_circle"
            if len(objects_from_scene(scene)) == 1
            and isinstance(objects_from_scene(scene)[0], Circle)
            else "converged_uniform_fdtd"
        )
        if accepted
        else None,
        "dt": base_record["dt"] if accepted else None,
        "reference_uncertainty_relative": uncertainty,
    }
    if accepted:
        field_path = report_path.with_suffix(".npz")
        temporary = field_path.with_name(field_path.stem + ".tmp.npz")
        np.savez_compressed(temporary, complex_far_field=accepted_field)
        os.replace(temporary, field_path)
    _write_json(report_path, report)
    return report
