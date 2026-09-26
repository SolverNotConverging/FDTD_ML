"""Restartable FDTD evaluation for supported continuous-geometry scenes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .analytic import cylinder_far_field
from .candidates_v2 import candidate_axes
from .curriculum_v2 import DOMAIN, objects_from_scene
from .geometry import Circle
from .grid import Grid
from .metrics import scattering_loss
from .monitor_v2 import widest_non_pml_monitor_bounds
from .scoring_v2 import relative_l2
from .solver_cuda import simulate_cuda
from .source import PlaneWave

FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
DURATIONS = (70e-9, 140e-9, 560e-9)
NUMERICAL_MODULES = (
    "analytic.py",
    "conformal.py",
    "compiled_cuda.py",
    "constants.py",
    "cuda_fdtd_kernels.cu",
    "cuda_fdtd_kernels.h",
    "_cuda_fdtd.pyx",
    "curriculum_v2.py",
    "geometry.py",
    "geometry_v2.py",
    "monitor_v2.py",
    "profiles_v2.py",
    "grid.py",
    "observables.py",
    "solver.py",
    "solver_cuda.py",
    "source.py",
    "simulation_v2.py",
)


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def numerical_source_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in NUMERICAL_MODULES
    }


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_arrays(path, **arrays):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _failure_status(error):
    message = str(error).lower()
    if any(
        fragment in message
        for fragment in (
            "split",
            "unresolved",
            "represented Ez nodes",
            "pec splits",
            "multiple pec cuts",
            "scatterer must lie",
            "monitor",
            "geometry",
            "open PEC edge",
        )
    ):
        return "geometry_incompatible"
    if "simulation requires" in message and "exceeding" in message:
        return "budget_limited"
    return "solver_error"


def evaluate_case(
    scene,
    cells,
    incidence_angle,
    policy,
    output,
    *,
    device="cuda:0",
    reference=None,
    reference_near_field=None,
    selected_axes=None,
    pml_thickness=0.12,
    material_samples=12,
    durations=DURATIONS,
    tail_tolerance=1e-5,
    monitor_bounds=None,
    source_hashes=None,
    field_sample_points=None,
):
    """Evaluate one case, saving raw fields and the independent numerical record."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = source_hashes or numerical_source_hashes()
    reference = None if reference is None else np.asarray(reference)
    reference_near_field = (
        None if reference_near_field is None else np.asarray(reference_near_field)
    )
    sample_points = (
        None if field_sample_points is None else np.asarray(field_sample_points, dtype=float)
    )
    if reference_near_field is not None and sample_points is None:
        raise ValueError("A near-field reference requires matching field sample points")
    fingerprint = _hash_json(
        {
            "scene": scene,
            "cells": cells,
            "angle": incidence_angle,
            "policy": policy,
            "pml": pml_thickness,
            "samples": material_samples,
            "tail_tolerance": tail_tolerance,
            "source": source_hashes,
            "durations": durations,
            "monitor_bounds": monitor_bounds,
            "selected_axes": [a.tolist() for a in selected_axes]
            if selected_axes is not None
            else None,
            "policy_sha256": hashlib.sha256(
                (Path(__file__).parent / "candidates_v2.py").read_bytes()
            ).hexdigest(),
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest()
            if reference is not None
            else None,
            "reference_near_field_sha256": hashlib.sha256(
                reference_near_field.tobytes()
            ).hexdigest()
            if reference_near_field is not None
            else None,
            "field_sample_points": sample_points.tolist() if sample_points is not None else None,
        }
    )
    record_path, arrays_path = output / "record.json", output / "spectra.npz"
    if record_path.exists() and arrays_path.exists():
        try:
            cached = json.loads(record_path.read_text())
            if cached.get("fingerprint") == fingerprint and hashlib.sha256(
                arrays_path.read_bytes()
            ).hexdigest() == cached.get("spectra_sha256"):
                return cached, True
        except (ValueError, OSError):
            pass
    record = {
        "schema_version": 2,
        "scene_id": scene["lineage_id"],
        "cells_x": cells,
        "cells_y": cells,
        "incidence_angle_rad": incidence_angle,
        "policy": policy,
        "fingerprint": fingerprint,
        "source_hashes": source_hashes,
        "status": "incomplete",
        "accepted": False,
    }
    try:
        if not np.isfinite(tail_tolerance) or tail_tolerance <= 0:
            raise ValueError("Positive finite field-tail tolerance is required")
        x, y, repair = candidate_axes(scene, cells, policy, selected_axes=selected_axes)
        grid = Grid(x, y, max_ratio=3)
        objects = objects_from_scene(scene)
        monitor = (
            widest_non_pml_monitor_bounds(grid, [obj.bounds for obj in objects], pml_thickness)
            if monitor_bounds is None
            else monitor_bounds
        )
        source = PlaneWave(1e9, 1e-9, 9e-9, angle=incidence_angle, origin=(DOMAIN / 2, DOMAIN / 2))
        if reference is None and len(objects) == 1 and isinstance(objects[0], Circle):
            obj = objects[0]
            reference = np.stack(
                [
                    cylinder_far_field(
                        obj.radius,
                        obj.material,
                        f,
                        ANGLES,
                        incidence_angle,
                        center=obj.center,
                        incident_origin=source.origin,
                    )
                    for f in FREQUENCIES
                ]
            )
        attempts = []
        result = None
        field = None
        for duration in durations:
            result = simulate_cuda(
                grid,
                objects,
                source,
                frequencies=FREQUENCIES,
                duration=duration,
                pml_thickness=pml_thickness,
                monitor_bounds=monitor,
                samples=material_samples,
                max_steps=2_000_000,
                pec_mode="conformal",
                device=device,
                dtype="float64",
                field_sample_points=sample_points,
            )
            field = result.monitor.normalized_far_field(ANGLES)
            tail = float(result.diagnostics["tail_peak_over_global_peak"])
            attempts.append(
                {
                    "duration_s": duration,
                    "Nt": result.diagnostics["Nt"],
                    "wall_seconds": result.diagnostics["wall_seconds"],
                    "tail_peak_over_global_peak": tail,
                }
            )
            if np.isfinite(field).all() and tail < tail_tolerance:
                break
        accepted = bool(np.isfinite(field).all() and tail < tail_tolerance)
        record.update(
            {
                "status": "accepted" if accepted else "unsettled",
                "accepted": accepted,
                "attempts": attempts,
                "uniform_repair_fraction": list(repair),
                **result.diagnostics,
            }
        )
        if reference is not None:
            record.update(scattering_loss(field, reference))
            width = 2 * np.pi * abs(field) ** 2
            width_ref = 2 * np.pi * abs(reference) ** 2
            record["complex_relative_l2"] = relative_l2(field, reference)
            record["width_relative_l2"] = relative_l2(width, width_ref)
        _atomic_arrays(
            arrays_path,
            x=x,
            y=y,
            frequencies=np.asarray(FREQUENCIES),
            angles=ANGLES,
            complex_far_field=field,
            **({"reference_complex_far_field": reference} if reference is not None else {}),
            **(
                {
                    "field_sample_points": result.fields["field_sample_points"],
                    "complex_near_field": (
                        result.fields["Ez_point_scattered_dft"]
                        + result.fields["Ez_point_incident_dft"]
                    )
                    / result.fields["Ez_point_incident_dft"],
                    "complex_scattered_near_field": result.fields["Ez_point_scattered_dft"]
                    / result.fields["Ez_point_incident_dft"],
                    "reference_complex_near_field": reference_near_field,
                }
                if sample_points is not None and reference_near_field is not None
                else (
                    {
                        "field_sample_points": result.fields["field_sample_points"],
                        "complex_near_field": (
                            result.fields["Ez_point_scattered_dft"]
                            + result.fields["Ez_point_incident_dft"]
                        )
                        / result.fields["Ez_point_incident_dft"],
                        "complex_scattered_near_field": result.fields["Ez_point_scattered_dft"]
                        / result.fields["Ez_point_incident_dft"],
                    }
                    if sample_points is not None
                    else {}
                )
            ),
        )
        if sample_points is not None:
            with np.load(arrays_path) as arrays:
                near_field = arrays["complex_near_field"].copy()
            record["near_field_point_count"] = int(len(sample_points))
            if reference_near_field is not None:
                record["near_field_relative_l2"] = relative_l2(near_field, reference_near_field)
        record["spectra_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    except (ValueError, RuntimeError, FloatingPointError) as error:
        record["status"] = _failure_status(error)
        record["error"] = f"{type(error).__name__}: {error}"
    _atomic_json(record_path, record)
    return record, False
