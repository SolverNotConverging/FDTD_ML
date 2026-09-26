"""Small GPU development campaign for the C8/C9 mesh-CNN proof of concept."""

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .campaign_v2 import COMPACT_TEACHER_NAMES, compact_conditions, scoring_fingerprint
from .curriculum_v2 import generate_compact_poc_scenes, generate_development_scenes
from .qualification_v2 import _write_json
from .scoring_v2 import rank_candidates, relative_l2
from .simulation_v2 import DURATIONS, evaluate_case, numerical_source_hashes
from .training_v2 import probe_microbatch

REFERENCE_LEVELS = (96, 128, 192, 256, 384)
REFERENCE_POLICY = {
    "protocol": "successive_uniform_compact_v1",
    "spatial_levels": REFERENCE_LEVELS,
    "complex_tolerance": 0.01,
    "width_tolerance": 0.02,
    "reference_tail_ratio": 1e-5,
    "independent_probe_tolerance": 0.01,
    "duration_probe_factor": 2.0,
    "pml_thickness_m": 0.12,
    "material_samples": 12,
    "durations_s": DURATIONS,
}
DEVELOPMENT_BUDGETS = (32, 48, 64, 96)
PROBE_ROLES = (
    "foundation",
    "heldout_silhouette_template",
    "separated_objects",
)


def _record_wall_seconds(record):
    attempts = record.get("attempts", ())
    return float(sum(item.get("wall_seconds", 0.0) for item in attempts)) or record.get(
        "wall_seconds"
    )


def _record_step_count(record):
    attempts = record.get("attempts", ())
    return int(sum(item.get("Nt", 0) for item in attempts)) or record.get("Nt")


def _raw_accuracy_headroom(candidate_rows):
    uniform = next(row for row in candidate_rows if row["name"] == "uniform")
    eligible = [
        row
        for row in candidate_rows
        if row.get("accepted")
        and row.get("joint_scattering_loss") is not None
        and np.isfinite(row["joint_scattering_loss"])
    ]
    if not uniform.get("accepted") or not eligible:
        return None, None
    best = min(eligible, key=lambda row: row["joint_scattering_loss"])
    gain = uniform["joint_scattering_loss"] / max(best["joint_scattering_loss"], 1e-30)
    return float(gain), best["name"]


def _load_field(directory, name="complex_far_field"):
    with np.load(Path(directory) / "spectra.npz") as arrays:
        return arrays[name].copy()


def _width(field):
    return 2 * np.pi * np.abs(field) ** 2


def _profile_scene(scene, output, device):
    output = Path(output)
    scene_root = output / "scenes" / scene["lineage_id"]
    observations = []
    reference = None
    selected_cells = None
    policy = REFERENCE_POLICY
    prior_field = None
    for cells in policy["spatial_levels"]:
        directory = scene_root / "references" / f"uniform_n{cells}"
        record, _ = evaluate_case(
            scene,
            cells,
            0.0,
            "uniform",
            directory,
            device=device,
            durations=policy["durations_s"],
            tail_tolerance=policy["reference_tail_ratio"],
            pml_thickness=policy["pml_thickness_m"],
            material_samples=policy["material_samples"],
        )
        row = {
            "cells": cells,
            "status": record["status"],
            "accepted": bool(record.get("accepted")),
            "wall_seconds": _record_wall_seconds(record),
            "Nt": _record_step_count(record),
            "dt": record.get("dt"),
            "tail": record.get("tail_peak_over_global_peak"),
            "monitor_bounds_m": record.get("monitor_bounds"),
        }
        observations.append(row)
        if not record.get("accepted"):
            break
        field = _load_field(directory)
        if prior_field is not None:
            complex_change = relative_l2(field, prior_field)
            width_change = relative_l2(_width(field), _width(prior_field))
            row["complex_change_from_previous"] = complex_change
            row["width_change_from_previous"] = width_change
            if (
                complex_change <= policy["complex_tolerance"]
                and width_change <= policy["width_tolerance"]
            ):
                selected_cells = cells
                reference = field
                with np.load(directory / "spectra.npz") as arrays:
                    if "reference_complex_far_field" in arrays:
                        reference = arrays["reference_complex_far_field"].copy()
                break
        prior_field = field

    result = {
        "scene_id": scene["lineage_id"],
        "family": scene["family"],
        "stage": scene["stage"],
        "development_role": scene.get("development_role"),
        "object_count": len(scene.get("objects", ())) or 1,
        "material": scene.get("material", {}).get("kind", "mixed"),
        "template_id": scene.get("template_id"),
        "monitor_policy": "per_grid_widest_non_pml_v1",
        "reference_observations": observations,
        "reference_accepted": selected_cells is not None,
        "reference_cells": selected_cells,
        "reference_uncertainty_relative": (
            max(
                observations[-1]["complex_change_from_previous"],
                observations[-1]["width_change_from_previous"],
            )
            if selected_cells is not None
            else None
        ),
        "teachers": [],
    }
    if reference is None:
        _write_json(scene_root / "profile.json", result)
        return result

    for cells in DEVELOPMENT_BUDGETS:
        candidate_rows = []
        for name in COMPACT_TEACHER_NAMES:
            directory = scene_root / "teachers" / f"n{cells}_{name}"
            record, _ = evaluate_case(
                scene,
                cells,
                0.0,
                name,
                directory,
                device=device,
                reference=reference,
                pml_thickness=policy["pml_thickness_m"],
                material_samples=policy["material_samples"],
            )
            candidate_rows.append(
                {
                    "name": name,
                    **record,
                    "wall_seconds": _record_wall_seconds(record),
                    "Nt": _record_step_count(record),
                }
            )
        try:
            ranked = rank_candidates(candidate_rows)
            best = ranked[0]["name"]
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            best = None
        gain, best_raw = _raw_accuracy_headroom(candidate_rows)
        result["teachers"].append(
            {
                "cells": cells,
                "candidates": [
                    {
                        key: record.get(key)
                        for key in (
                            "name",
                            "status",
                            "accepted",
                            "joint_scattering_loss",
                            "complex_relative_l2",
                            "width_relative_l2",
                            "dt",
                            "Nt",
                            "wall_seconds",
                            "peak_cuda_memory_bytes",
                            "monitor_bounds",
                        )
                    }
                    for record in candidate_rows
                ],
                "best_teacher": best,
                "best_raw_accuracy_candidate": best_raw,
                "uniform_over_best_raw_accuracy_ratio": gain,
            }
        )
    _write_json(scene_root / "profile.json", result)
    return result


def _probe_reference(scene, scene_root, device, reference, base_record, policy):
    base_duration = base_record["attempts"][-1]["duration_s"]
    longer = policy["duration_probe_factor"] * base_duration
    a, b, c, d = base_record["monitor_bounds"]
    specs = {
        "duration": {"durations": (longer,)},
        "quadrature": {
            "durations": (base_duration,),
            "material_samples": 24,
        },
        "contour": {
            "durations": (base_duration,),
            "monitor_bounds": (a + 0.015, b - 0.015, c + 0.015, d - 0.015),
        },
    }
    output = {}
    for name, overrides in specs.items():
        kwargs = {
            "durations": (base_duration,),
            "tail_tolerance": policy["reference_tail_ratio"],
            "pml_thickness": policy["pml_thickness_m"],
            "material_samples": policy["material_samples"],
            "monitor_bounds": base_record["monitor_bounds"],
            **overrides,
        }
        directory = scene_root / "reference_probes" / name
        record, _ = evaluate_case(
            scene,
            int(base_record["cells_x"]),
            float(base_record["incidence_angle_rad"]),
            "uniform",
            directory,
            device=device,
            reference=reference,
            **kwargs,
        )
        field = _load_field(directory) if record.get("accepted") else None
        output[name] = {
            "status": record["status"],
            "complex_change": relative_l2(field, reference) if field is not None else None,
            "width_change": relative_l2(_width(field), _width(reference))
            if field is not None
            else None,
            "wall_seconds": record.get("wall_seconds"),
        }
    from .candidates_v2 import candidate_axes
    from .curriculum_v2 import objects_from_scene
    from .grid import Grid
    from .monitor_v2 import widest_non_pml_monitor_bounds

    cells = int(base_record["cells_x"])
    axis, _, _ = candidate_axes(scene, cells, "uniform")
    objects = objects_from_scene(scene)
    common_monitor = widest_non_pml_monitor_bounds(
        Grid(axis, axis, max_ratio=3), [obj.bounds for obj in objects], 0.15
    )
    pml_records = []
    pml_fields = []
    for name, pml in (("pml_control", policy["pml_thickness_m"]), ("pml", 0.15)):
        directory = scene_root / "reference_probes" / name
        record, _ = evaluate_case(
            scene,
            cells,
            float(base_record["incidence_angle_rad"]),
            "uniform",
            directory,
            device=device,
            durations=(base_duration,),
            tail_tolerance=policy["reference_tail_ratio"],
            pml_thickness=pml,
            material_samples=policy["material_samples"],
            monitor_bounds=common_monitor,
        )
        pml_records.append(record)
        pml_fields.append(_load_field(directory) if record.get("accepted") else None)
    control, altered = pml_fields
    output["pml"] = {
        "status": "accepted" if all(row.get("accepted") for row in pml_records) else "incomplete",
        "complex_change": relative_l2(altered, control)
        if control is not None and altered is not None
        else None,
        "width_change": relative_l2(_width(altered), _width(control))
        if control is not None and altered is not None
        else None,
        "contour_control_complex_change": relative_l2(control, reference)
        if control is not None
        else None,
        "contour_control_width_change": relative_l2(_width(control), _width(reference))
        if control is not None
        else None,
        "wall_seconds": sum(row.get("wall_seconds", 0.0) for row in pml_records),
        "common_monitor_bounds_m": common_monitor,
    }
    return output


def _development_probes(scenes, rows, output, device):
    by_id = {scene["lineage_id"]: scene for scene in scenes}
    selected = []
    for role in PROBE_ROLES:
        selected.append(next(row for row in rows if row.get("development_role") == role))
    probes = {}
    for row in selected:
        scene = by_id[row["scene_id"]]
        scene_root = Path(output) / "scenes" / scene["lineage_id"]
        cells = row["reference_cells"]
        base_directory = scene_root / "references" / f"uniform_n{cells}"
        with np.load(base_directory / "spectra.npz") as arrays:
            reference = (
                arrays["reference_complex_far_field"].copy()
                if "reference_complex_far_field" in arrays
                else arrays["complex_far_field"].copy()
            )
        base_record = json.loads((base_directory / "record.json").read_text())
        probes[row["scene_id"]] = _probe_reference(
            scene, scene_root, device, reference, base_record, REFERENCE_POLICY
        )
    return probes


def _representative_costs(rows):
    """Return empirical per-update teacher costs and per-angle reference costs."""
    samples = {}
    reference_durations = {}
    for row in rows:
        group = (
            "silhouette"
            if row["stage"] == "C8"
            else ("collection" if row["stage"] in {"C3", "C5"} else "foundation")
        )
        for teacher in row["teachers"]:
            for candidate in teacher["candidates"]:
                if not candidate.get("accepted"):
                    continue
                key = (group, teacher["cells"], candidate["name"])
                denominator = (candidate.get("Nt") or 0) * teacher["cells"] ** 2
                if denominator and candidate.get("wall_seconds"):
                    samples.setdefault(key, []).append(candidate["wall_seconds"] / denominator)
        observed_reference_seconds = sum(
            observation.get("wall_seconds", 0.0)
            for observation in row.get("reference_observations", ())
        )
        if observed_reference_seconds > 0:
            reference_durations.setdefault(group, []).append(observed_reference_seconds)
    costs = {key: statistics.median(values) for key, values in samples.items() if values}
    for group, values in reference_durations.items():
        costs[("reference_seconds_per_scene", group)] = statistics.median(values)
    return costs


def _estimate_campaign_hours(rows, scenes, costs, device_count, train_benchmark, seed):
    scene_by_id = {scene["lineage_id"]: scene for scene in scenes}
    compact_scenes = generate_compact_poc_scenes(seed=seed)
    condition_map, validation_map, test_map = compact_conditions(compact_scenes, seed)
    phase_gpu_seconds = {"data": 0.0, "validation": 0.0, "test": 0.0}
    missing = []

    def stage_group(scene):
        return (
            "silhouette"
            if scene["stage"] == "C8"
            else ("collection" if scene["stage"] in {"C3", "C5", "C9"} else "foundation")
        )

    def scene_scales(scene, group):
        objects = scene.get("objects", ()) or (scene,)
        object_scale = max(1.0, len(objects) / 3.0) if group == "collection" else 1.0
        pec_fraction = sum(obj.get("material", {}).get("kind") == "pec" for obj in objects) / len(
            objects
        )
        return object_scale, 1.0 + 0.25 * pec_fraction

    def add(scene, cells, method, *, phase, multiplicity=1):
        group = stage_group(scene)
        policy = "interface" if method == "learned" else method
        per_update = costs.get((group, cells, policy))
        if per_update is None:
            missing.append((scene["stage"], cells, method))
            return
        observations = [
            candidate
            for row in rows
            if stage_group({"stage": row["stage"]}) == group
            for teacher in row.get("teachers", ())
            if teacher["cells"] == cells
            for candidate in teacher["candidates"]
            if candidate.get("accepted") and candidate.get("Nt")
        ]
        if not observations:
            missing.append((scene["stage"], cells, method))
            return
        steps = max(candidate["Nt"] for candidate in observations)
        object_scale, material_scale = scene_scales(scene, group)
        phase_gpu_seconds[phase] += (
            per_update * cells**2 * steps * multiplicity * object_scale * material_scale
        )

    def add_reference(scene, *, phase, multiplicity=1):
        group = stage_group(scene)
        seconds = costs.get(("reference_seconds_per_scene", group))
        if seconds is None:
            missing.append((scene["stage"], "reference", phase))
            return
        object_scale, material_scale = scene_scales(scene, group)
        phase_gpu_seconds[phase] += seconds * multiplicity * object_scale * material_scale

    for lineage, conditions in condition_map.items():
        scene = scene_by_id[lineage]
        for angle_index in {condition["angle_index"] for condition in conditions}:
            add_reference(scene, phase="data")
        for condition in conditions:
            for method in COMPACT_TEACHER_NAMES:
                add(scene, condition["cells"], method, phase="data")

    for lineage, conditions in validation_map.items():
        scene = scene_by_id[lineage]
        for condition in conditions:
            for _ in range(3):
                add(scene, condition["cells"], "learned", phase="validation")

    for lineage, conditions in test_map.items():
        scene = scene_by_id[lineage]
        for angle_index in {condition["angle_index"] for condition in conditions}:
            add_reference(scene, phase="test")
        for condition in conditions:
            for method in ("uniform", "interface", "learned"):
                add(scene, condition["cells"], method, phase="test")

    training_step = train_benchmark.get("step_seconds")
    microbatch = train_benchmark.get("microbatch")
    training_hours = None
    if training_step and microbatch:
        micro_steps = 60 * 4 * (32 // microbatch)
        train_seconds = training_step * micro_steps * 1.5
        validation_forward_seconds = (training_step / 3) * (16 // microbatch) * 60
        training_hours = (train_seconds + validation_forward_seconds) / 3600
    data_workers = max(1, min(4, device_count))
    fdt_wall_by_phase = {
        "data": phase_gpu_seconds["data"] / data_workers / 3600,
        "validation": phase_gpu_seconds["validation"] / 3600,
        "test": phase_gpu_seconds["test"] / 3600,
    }
    fdt_wall_hours = sum(fdt_wall_by_phase.values())
    estimate = (fdt_wall_hours + (training_hours or 0.0) + 0.5) * 1.35
    return {
        "estimated_compact_campaign_hours": estimate,
        "estimated_fdt_gpu_hours": sum(phase_gpu_seconds.values()) / 3600,
        "estimated_fdt_wall_hours_at_profiled_gpu_count": fdt_wall_hours,
        "estimated_fdt_wall_hours_by_phase": fdt_wall_by_phase,
        "estimated_data_worker_gpu_count": data_workers,
        "estimated_training_hours": training_hours,
        "profiled_gpu_count": device_count,
        "cost_model_margin": 1.35,
        "missing_cost_cells": [list(item) for item in sorted(set(missing), key=repr)],
        "cost_model_basis": (
            "Measured development-case seconds per cell-update plus reference runtimes; "
            "four data workers and one GPU for validation/test; "
            "60-epoch upper training allowance; 35% workload margin."
        ),
    }


def run_compact_profile(
    output,
    *,
    devices=None,
    seed=20260924,
    max_campaign_hours=24.0,
):
    """Measure eight development lineages and gate the first bounded campaign."""
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if devices is None:
        devices = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    devices = tuple(devices)
    if any(not str(device).startswith("cuda") for device in devices):
        raise ValueError("Development FDTD profiling requires the compiled CUDA kernel")
    scenes = generate_development_scenes(seed)
    if not torch.cuda.is_available() or not devices:
        result = {
            "schema_version": 1,
            "profile": "compact_c8_c9_development_v1",
            "seed": seed,
            "status": "gpu_unavailable",
            "gate": "gpu_unavailable",
            "estimated_compact_campaign_hours": None,
            "required_gpu_count": 1,
            "available_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "scenes": [
                {"scene_id": scene["lineage_id"], "stage": scene["stage"]} for scene in scenes
            ],
            "note": (
                "Development FDTD timings and a GPU training-step benchmark are required "
                "before freezing a compact campaign manifest."
            ),
        }
        _write_json(output / "compact_profile.json", result)
        return result

    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=get_context("spawn")) as pool:
        future_to_scene = {
            pool.submit(_profile_scene, scene, output, devices[index % len(devices)]): scene
            for index, scene in enumerate(scenes)
        }
        for future in as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append(
                    {
                        "scene_id": scene["lineage_id"],
                        "stage": scene["stage"],
                        "status": "profile_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    rows.sort(key=lambda row: row["scene_id"])

    benchmark = {}
    try:
        selected_microbatch, probe = probe_microbatch(64, device=devices[0])
        measured = next(row for row in probe if row["microbatch"] == selected_microbatch)
        benchmark = {
            "microbatch": selected_microbatch,
            "step_seconds": measured["step_seconds"],
            "probe": probe,
        }
    except (RuntimeError, ValueError) as error:
        benchmark = {"error": f"{type(error).__name__}: {error}"}

    ready_rows = [row for row in rows if row.get("reference_accepted")]
    try:
        probes = _development_probes(scenes, ready_rows, output, devices[0])
    except (StopIteration, OSError, ValueError, RuntimeError) as error:
        probes = {"profile_error": f"{type(error).__name__}: {error}"}

    costs = _representative_costs(rows)
    projection = _estimate_campaign_hours(
        rows,
        generate_compact_poc_scenes(seed=seed),
        costs,
        len(devices),
        benchmark,
        seed,
    )
    # Optimizing every train/validation condition adds 96 physics evaluations.
    # A high observed candidate cost quantile gives a conservative first sizing
    # estimate; the frozen manifest requires this cost to be included.
    search_wall_samples = [
        candidate["wall_seconds"]
        for row in rows
        for teacher in row.get("teachers", ())
        for candidate in teacher["candidates"]
        if candidate.get("wall_seconds") and candidate["status"] in ("accepted", "unsettled")
    ]
    search_conditions, _, _ = compact_conditions(generate_compact_poc_scenes(seed=seed), seed)
    optimization_conditions = sum(len(conditions) for conditions in search_conditions.values())
    search_candidate_seconds = (
        float(np.quantile(search_wall_samples, 0.95)) if search_wall_samples else None
    )
    search_wall_hours = (
        optimization_conditions
        * 96
        * search_candidate_seconds
        * projection["cost_model_margin"]
        / (3600 * len(devices))
        if search_candidate_seconds and devices
        else None
    )
    projection["teacher_search_calibration"] = {
        "method": "96_evaluation_smooth_density_differential_evolution",
        "train_validation_conditions": optimization_conditions,
        "observed_candidate_wall_seconds_p95": search_candidate_seconds,
        "estimated_teacher_search_wall_hours": search_wall_hours,
        "basis": "95th percentile of fixed-policy development candidate wall times with cost margin",
    }
    if search_wall_hours is not None:
        projection["estimated_compact_campaign_hours"] += search_wall_hours
        projection["estimated_fdt_wall_hours_at_profiled_gpu_count"] += search_wall_hours
        projection["estimated_fdt_gpu_hours"] += search_wall_hours * len(devices)
    accepted_references = sum(row.get("reference_accepted", False) for row in rows)
    teacher_gains = [
        item["uniform_over_best_raw_accuracy_ratio"]
        for row in rows
        for item in row.get("teachers", ())
        if item.get("uniform_over_best_raw_accuracy_ratio") is not None
    ]
    positive_development = any(value >= 1.05 for value in teacher_gains)
    probe_items = [
        item
        for scene_probe in probes.values()
        if isinstance(scene_probe, dict)
        for item in scene_probe.values()
        if isinstance(item, dict)
    ]
    probes_pass = (
        bool(probe_items)
        and len(probes) == len(PROBE_ROLES)
        and all(
            item.get("status") == "accepted"
            and item.get("complex_change") is not None
            and item.get("width_change") is not None
            and item["complex_change"] <= REFERENCE_POLICY["independent_probe_tolerance"]
            and item["width_change"] <= REFERENCE_POLICY["independent_probe_tolerance"]
            and item.get("contour_control_complex_change", 0.0)
            <= REFERENCE_POLICY["independent_probe_tolerance"]
            and item.get("contour_control_width_change", 0.0)
            <= REFERENCE_POLICY["independent_probe_tolerance"]
            for item in probe_items
        )
    )
    complete = len(rows) == 8 and accepted_references == 8
    estimate = projection["estimated_compact_campaign_hours"]
    ready = bool(
        complete
        and positive_development
        and probes_pass
        and benchmark.get("microbatch")
        and not projection["missing_cost_cells"]
        and search_wall_hours is not None
        and estimate <= max_campaign_hours
    )
    result = {
        "schema_version": 1,
        "profile": "compact_c8_c9_development_v1",
        "seed": seed,
        "status": "ready_for_compact_campaign" if ready else "report_before_campaign",
        "gate": "ready_for_compact_campaign" if ready else "report_before_campaign",
        "elapsed_seconds": time.time() - started,
        "gpu_devices": list(devices),
        "development_scene_count": len(rows),
        "qualified_reference_count": accepted_references,
        "reference_policy": REFERENCE_POLICY,
        "teacher_candidates": COMPACT_TEACHER_NAMES,
        "development_teacher_gain_ratios": teacher_gains,
        "positive_development_teacher_gain": positive_development,
        "reference_sensitivity_probes": probes,
        "reference_probes_pass": probes_pass,
        "training_benchmark": benchmark,
        **projection,
        "max_campaign_hours": max_campaign_hours,
        "numerical_source_hashes": numerical_source_hashes(),
        "scoring_fingerprint": scoring_fingerprint(),
        "rows": rows,
    }
    _write_json(output / "compact_profile.json", result)
    return result
