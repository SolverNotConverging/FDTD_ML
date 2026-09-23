#!/usr/bin/env python3
"""Check sparse-pilot mesh geometry and exact time-step costs before GPU runs."""

import argparse
import hashlib
import inspect
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scattermesh import simulate_cuda, sparse_scene_objects
from scattermesh.conformal import CutCellPEC
from scattermesh.constants import C0
from scattermesh.generalization import widest_non_pml_monitor_bounds
from scattermesh.geometry import split_material_objects
from scattermesh.observables import Contour
from run_sparse_pair_headroom_pilot import definitions, load_config, make_grid

ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def preflight(config_path):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    signature = inspect.signature(simulate_cuda)
    safety = signature.parameters["safety"].default
    max_steps = signature.parameters["max_steps"].default
    durations = config["duration_schedule_s"]
    records = []
    for phase in ("references", "candidates"):
        for definition in definitions(config, phase):
            scene = definition["scene"]
            policy = definition["policy"]
            row = {
                "case_id": definition["case_id"],
                "phase": phase,
                "scene_id": scene["scene_id"],
                "scene_family_id": scene.get("family_id"),
                "split": scene.get("split"),
                "pec_circle_count": sum(
                    obj["shape"] == "circle" and obj["material"].get("kind") == "pec"
                    for obj in scene["objects"]
                ),
                "cells": definition["cells"],
                "policy": policy["name"],
            }
            try:
                grid = make_grid(config, definition)
                objects = sparse_scene_objects(scene)
                monitor = widest_non_pml_monitor_bounds(
                    grid, [obj.bounds for obj in objects], config["pml_thickness_m"]
                )
                Contour(grid, monitor)
                pec, _ = split_material_objects(objects)
                cut = CutCellPEC(grid, pec, config.get("pec_mode", "conformal")) if pec else None
                dx, dy = np.diff(grid.x), np.diff(grid.y)
                grid_dt = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
                pec_dt = cut.stable_time_step() if cut else None
                dt = safety * min(grid_dt, pec_dt) if cut else safety * grid_dt
                nt = [math.ceil(duration / dt) for duration in durations]
                row.update({
                    "mesh_feasible": True,
                    "monitor_bounds_m": list(monitor),
                    "grid_stable_dt_s": float(grid_dt),
                    "pec_stable_dt_s": float(pec_dt) if cut else None,
                    "time_step_s": float(dt),
                    "minimum_pec_cut_fraction": float(cut.minimum_fraction) if cut else None,
                    "required_steps": nt,
                    "within_step_limit": [count <= max_steps for count in nt],
                })
            except ValueError as error:
                row.update({"mesh_feasible": False, "error": str(error)})
            records.append(row)

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["phase"], row["scene_family_id"])].append(row)
    summary = {}
    for (phase, family), rows in sorted(grouped.items()):
        feasible = [row for row in rows if row["mesh_feasible"]]
        summary[f"{phase}:{family}"] = {
            "case_count": len(rows),
            "mesh_infeasible": len(rows) - len(feasible),
            "base_over_step_limit": sum(not row["within_step_limit"][0] for row in feasible),
            "first_retry_over_step_limit": sum(not row["within_step_limit"][1] for row in feasible),
            "last_retry_over_step_limit": sum(not row["within_step_limit"][-1] for row in feasible),
            "maximum_base_steps": max((row["required_steps"][0] for row in feasible), default=None),
            "minimum_pec_cut_fraction": min(
                (row["minimum_pec_cut_fraction"] for row in feasible
                 if row["minimum_pec_cut_fraction"] is not None),
                default=None,
            ),
        }
    return {
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in (
                *sorted((ROOT / "src/scattermesh").glob("*.py")),
                ROOT / "scripts/run_sparse_pair_headroom_pilot.py",
            )
        },
        "preflight_script_sha256": _sha256_file(__file__),
        "safety": safety,
        "max_steps": max_steps,
        "duration_schedule_s": durations,
        "case_count": len(records),
        "initial_run_ready": all(
            row["mesh_feasible"] and row["within_step_limit"][0] for row in records
        ),
        "summary": summary,
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sparse_cluster_pilot_32.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runs/sparse_cluster_pilot_32/cutcell_preflight.json",
    )
    args = parser.parse_args()
    result = preflight(args.config)
    _atomic_json(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "case_count": result["case_count"],
        "initial_run_ready": result["initial_run_ready"],
        "summary": result["summary"],
    }, indent=2))
    if not result["initial_run_ready"]:
        raise SystemExit("Some initial GPU runs cannot start; inspect the preflight report")


if __name__ == "__main__":
    main()
