"""Measured 16-scene sizing gate before committing GPU time to bulk labels."""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .candidates_v2 import candidate_axes
from .conformal import CutCellPEC
from .constants import C0
from .curriculum_v2 import FAMILIES, generate_lineages, object_from_scene
from .geometry import PEC
from .grid import Grid
from .simulation_v2 import evaluate_case


def estimated_steps(scene, cells, policy="uniform", duration=70e-9):
    x, y, _ = candidate_axes(scene, cells, policy)
    grid = Grid(x, y, max_ratio=3)
    spacing_x, spacing_y = np.diff(x), np.diff(y)
    grid_dt = 1 / (C0 * np.sqrt(spacing_x.min() ** -2 + spacing_y.min() ** -2))
    obj = object_from_scene(scene)
    cut = CutCellPEC(grid, [obj]) if isinstance(obj.material, PEC) else None
    dt = 0.9 * min(grid_dt, cut.stable_time_step() if cut else grid_dt)
    return {
        "dt": dt,
        "steps_70ns": math.ceil(duration / dt),
        "steps_140ns": math.ceil(140e-9 / dt),
        "steps_560ns": math.ceil(560e-9 / dt),
    }


def _one_profile(scene, output, device, max_profile_steps):
    data = {
        "scene_id": scene["lineage_id"],
        "family": scene["family"],
        "material": scene["material"]["kind"],
        "estimates": {},
        "measurements": {},
    }
    for cells, policy in (
        (32, "uniform"),
        (32, "interface"),
        (48, "interface"),
        (64, "interface"),
        (96, "interface"),
        (192, "uniform"),
        (256, "uniform"),
        (384, "uniform"),
    ):
        label = f"n{cells}_{policy}"
        try:
            data["estimates"][label] = estimated_steps(scene, cells, policy)
        except (ValueError, RuntimeError) as error:
            data["estimates"][label] = {"status": "geometry_incompatible", "error": str(error)}
    for cells in (32, 192):
        estimate = data["estimates"][f"n{cells}_uniform"]
        if estimate.get("steps_70ns", float("inf")) > max_profile_steps:
            data["measurements"][str(cells)] = {"status": "profile_step_cap"}
            continue
        started = time.perf_counter()
        record, _ = evaluate_case(
            scene,
            cells,
            0.0,
            "uniform",
            Path(output) / "cases" / f"{scene['lineage_id']}_n{cells}",
            device=device,
            durations=(70e-9,),
        )
        data["measurements"][str(cells)] = {
            "status": record["status"],
            "wall_seconds": time.perf_counter() - started,
            "steps": record.get("Nt"),
            "tail": record.get("tail_peak_over_global_peak"),
        }
    return data


def _project_row(row, factor, *, duration_factor=1):
    estimates = row["estimates"]
    measurement = row["measurements"].get("192", {})
    if measurement.get("steps") and measurement.get("wall_seconds"):
        seconds_per_cell_step = measurement["wall_seconds"] / (measurement["steps"] * 192**2)
    else:
        seconds_per_cell_step = factor * (2 if row["material"] == "pec" else 1)

    def seconds(cells, policy, repeats=1):
        item = estimates.get(f"n{cells}_{policy}", {})
        steps = item.get("steps_140ns")
        if steps is None or steps > 2_000_000:
            return None
        return seconds_per_cell_step * cells**2 * steps * repeats * duration_factor

    reference_parts = [seconds(c, "uniform") for c in (192, 256, 384)]
    reference_parts.append(seconds(384, "uniform", 4))
    candidate_parts = [seconds(c, "interface", 11) for c in (32, 48, 64, 96)]
    candidate_parts += [
        seconds(c, "uniform") for c in (32, 48, 64, 96) if f"n{c}_uniform" in estimates
    ]
    # Uniform low-budget step count is close to the measured grid-CFL ratio.
    candidate_parts += [seconds(c, "interface") for c in (48, 64, 96)]
    if any(value is None for value in reference_parts + candidate_parts):
        return {"status": "incompatible_or_step_cap", "gpu_seconds_per_scene": None}
    return {
        "status": "projected",
        "gpu_seconds_per_scene": 2 * sum(reference_parts + candidate_parts),
        "seconds_per_cell_step": seconds_per_cell_step,
    }


def run_profile(
    output,
    *,
    devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    max_profile_steps=50_000,
    seed=20260923,
):
    """Profile 16 shape/material exemplars, then decide 256, 128, or stop."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_scenes = generate_lineages(128, seed=seed)
    scenes = [
        next(
            scene
            for scene in all_scenes
            if scene["family"] == family and scene["material"]["kind"] == material
        )
        for family in FAMILIES
        for material in ("dielectric", "pec")
    ]
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=len(devices)) as pool:
        future_map = {
            pool.submit(
                _one_profile, scene, output, devices[index % len(devices)], max_profile_steps
            ): scene["lineage_id"]
            for index, scene in enumerate(scenes)
        }
        for future in as_completed(future_map):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append(
                    {
                        "scene_id": future_map[future],
                        "status": "profile_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    measured = [
        row["measurements"]["192"]
        for row in rows
        if "measurements" in row and row["measurements"].get("192", {}).get("steps")
    ]
    factors = [row["wall_seconds"] / (row["steps"] * 192**2) for row in measured]
    factor = float(np.median(factors)) if factors else None
    projections = (
        [_project_row(row, factor) for row in rows if "estimates" in row] if factor else []
    )
    compatible = [
        (scene, projection)
        for scene, projection in zip(rows, projections)
        if projection["gpu_seconds_per_scene"] is not None
    ]
    complete = len(compatible) == 16 and len(rows) == 16

    # Each family has 12 dielectric and 4 PEC lineages for a 128-lineage corpus.
    def hours(count, *, partial):
        if not partial and not complete:
            return None
        per_family = count // len(FAMILIES)
        total = sum(
            projection["gpu_seconds_per_scene"]
            * per_family
            * (0.75 if scene["material"] == "dielectric" else 0.25)
            for scene, projection in compatible
        )
        return total / len(devices) / 3600

    estimates = {str(count): hours(count, partial=False) for count in (128, 256)}
    partial_projections = {str(count): hours(count, partial=True) for count in (128, 256)}
    choice = (
        256
        if estimates["256"] is not None and estimates["256"] <= 14
        else (128 if estimates["128"] is not None and estimates["128"] <= 14 else None)
    )
    result = {
        "schema_version": 2,
        "profile_scene_count": len(scenes),
        "completed_rows": len(rows),
        "elapsed_seconds": time.time() - started,
        "gpu_devices": list(devices),
        "median_seconds_per_cell_step": factor,
        "projected_data_hours_by_lineages": estimates,
        "partial_projected_data_hours_by_lineages": partial_projections,
        "compatible_profile_rows": len(compatible),
        "data_allocation_hours": 14,
        "selected_lineages": choice,
        "gate": "ready_for_bulk" if choice else "report_before_bulk",
        "note": "Projection assumes 140 ns teacher duration and four 384-cell independent probes; unresolved PEC steps or incomplete profile block launch.",
        "rows": rows,
        "projections": projections,
    }
    temporary = output / "profile.tmp.json"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output / "profile.json")
    return result
