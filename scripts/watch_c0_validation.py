"""Evaluate the three best C0 checkpoints on qualified validation FDTD cases."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scattermesh.campaign_v2 import atomic_json
from scattermesh.evaluation_v2 import load_checkpoint, predict_axes
from scattermesh.simulation_v2 import evaluate_case


def _paths(run):
    run = Path(run).resolve()
    return run, run / "validation", run / "dataset" / "dataset.json"


def _validation_examples(dataset_path):
    metadata = json.loads(dataset_path.read_text())
    return [example for example in metadata["examples"] if example["split"] == "validation"]


def evaluate(run, checkpoint_path, device):
    run, output, dataset_path = _paths(run)
    checkpoint_path = Path(checkpoint_path).resolve()
    name = checkpoint_path.stem
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    manifest = json.loads((run / "manifest.json").read_text())
    examples = _validation_examples(dataset_path)
    rows = []
    for example in examples:
        scene = example["scene"]
        cells = int(example["cells_x"])
        if cells != int(example["cells_y"]):
            raise ValueError("The C0 validation protocol requires square mesh budgets")
        angle = float(example["incidence_angle_rad"])
        angle_index = min(range(len(manifest["incidence_angles"])), key=lambda i: abs(manifest["incidence_angles"][i] - angle))
        if abs(manifest["incidence_angles"][angle_index] - angle) > 1e-12:
            raise ValueError("Validation angle is absent from the frozen manifest")
        qualification = run / "data" / "qualifications" / f"{scene['lineage_id']}_a{angle_index}.json"
        reference_path = qualification.with_suffix(".npz")
        qualified = json.loads(qualification.read_text())
        if not qualified.get("accepted") or not reference_path.is_file():
            raise ValueError(f"Validation example lost its qualified reference: {example['sample_id']}")
        with np.load(reference_path) as arrays:
            reference = arrays["complex_far_field"].copy()
        baseline_path = run / "data" / "cases" / f"{example['sample_id']}_uniform" / "record.json"
        uniform = json.loads(baseline_path.read_text())
        if not uniform.get("accepted") or uniform.get("joint_scattering_loss") is None:
            raise ValueError(f"Validation example lost its uniform baseline: {example['sample_id']}")
        axes, repairs = predict_axes(model, scene, cells, angle, example["frequencies_hz"], device)
        learned, _ = evaluate_case(
            scene, cells, angle, "learned", destination / "cases" / example["sample_id"],
            device=device, reference=reference, selected_axes=axes,
        )
        row = {
            "sample_id": example["sample_id"],
            "lineage_id": scene["lineage_id"],
            "family": scene["family"],
            "material_kind": scene["material"]["kind"],
            "cells": cells,
            "status": learned["status"],
            "accepted": bool(learned.get("accepted")),
            "learned_raw_accuracy_loss": learned.get("joint_scattering_loss") if learned.get("accepted") else None,
            "uniform_raw_accuracy_loss": uniform["joint_scattering_loss"],
            "reference_uncertainty_relative": qualified.get("reference_uncertainty_relative"),
            "projection_repair_fractions": list(repairs),
            "learned_dt": learned.get("dt"),
            "uniform_dt": uniform.get("dt"),
        }
        rows.append(row)
        atomic_json(destination / "summary.json", {
            "split": "validation", "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"], "requested_conditions": len(examples),
            "valid_conditions": sum(item["accepted"] for item in rows),
            "terminal": False, "rows": rows,
        })
    valid = [row for row in rows if row["accepted"]]
    summary = {
        "schema_version": 1,
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_profile_loss": float(checkpoint["validation_profile_loss"]),
        "requested_conditions": len(examples),
        "valid_conditions": len(valid),
        "mean_learned_raw_accuracy_loss": float(np.mean([row["learned_raw_accuracy_loss"] for row in valid])) if valid else None,
        "mean_uniform_raw_accuracy_loss": float(np.mean([row["uniform_raw_accuracy_loss"] for row in valid])) if valid else None,
        "median_uniform_over_learned_ratio": float(np.median([row["uniform_raw_accuracy_loss"] / max(row["learned_raw_accuracy_loss"], 1e-30) for row in valid])) if valid else None,
        "terminal": True,
        "rows": rows,
    }
    atomic_json(destination / "summary.json", summary)
    return summary


def select(run):
    run, output, _ = _paths(run)
    summaries = [json.loads(path.read_text()) for path in sorted(output.glob("checkpoint_epoch_*/summary.json"))]
    status_path = output / "selection_status.json"
    if len(summaries) != 3 or any(
        not item.get("terminal") or item["valid_conditions"] != item["requested_conditions"]
        for item in summaries
    ):
        result = {
            "status": "validation_incomplete",
            "checkpoint_summaries": [
                {"checkpoint": item["checkpoint"], "valid_conditions": item["valid_conditions"], "requested_conditions": item["requested_conditions"]}
                for item in summaries
            ],
        }
        atomic_json(status_path, result)
        return result
    selected = min(summaries, key=lambda item: item["mean_learned_raw_accuracy_loss"])
    selection = {
        "schema_version": 1,
        "selection_metric": "mean_raw_physical_accuracy_on_qualified_validation_conditions",
        "checkpoint": selected["checkpoint"],
        "checkpoint_epoch": selected["checkpoint_epoch"],
        "validation_raw_accuracy": selected["mean_learned_raw_accuracy_loss"],
        "validation_conditions": selected["valid_conditions"],
        "comparison": [
            {
                "checkpoint": item["checkpoint"],
                "checkpoint_epoch": item["checkpoint_epoch"],
                "mean_raw_accuracy": item["mean_learned_raw_accuracy_loss"],
                "median_uniform_over_learned_ratio": item["median_uniform_over_learned_ratio"],
            }
            for item in summaries
        ],
    }
    frozen_path = run / "large_cnn" / "frozen_selection.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != selection:
        raise ValueError("Frozen C0 checkpoint selection already differs")
    atomic_json(frozen_path, selection)
    result = {"status": "checkpoint_frozen", **selection}
    atomic_json(status_path, result)
    return result


def watch(run):
    run, output, _ = _paths(run)
    launch = json.loads((run / "launch.json").read_text())
    deadline = float(launch["deadline_epoch_s"])
    finalization_path = run / "finalization.json"
    status_path = output / "selection_status.json"
    project_status = run.parents[1] / "project_execution_status.json"
    while time.time() < deadline:
        if finalization_path.exists():
            finalization = json.loads(finalization_path.read_text())
            if finalization.get("status") == "training_finished":
                break
            if finalization.get("status") in ("insufficient_qualified_labels", "acquisition_deadline"):
                result = {"status": "training_unavailable", "reason": finalization["status"]}
                atomic_json(status_path, result)
                return result
        time.sleep(30)
    else:
        result = {"status": "training_deadline"}
        atomic_json(status_path, result)
        return result
    if finalization["result"].get("deadline_reached"):
        result = {"status": "training_deadline"}
        atomic_json(status_path, result)
        return result
    checkpoints = sorted((run / "large_cnn").glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) != 3:
        result = {"status": "missing_top_three_checkpoints", "found": len(checkpoints)}
        atomic_json(status_path, result)
        return result
    status = {"status": "validation_running", "checkpoints": [str(path) for path in checkpoints]}
    atomic_json(status_path, status)
    atomic_json(project_status, status | {"authoritative_plan": "IMPLEMENTATION_PLAN.md"})
    processes = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        log = output / f"{checkpoint.stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab", buffering=0)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "evaluate", str(run), str(checkpoint), f"cuda:{index}"],
            cwd=run.parents[2], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes.append((process, handle))
    return_codes = [process.wait() for process, _ in processes]
    for _, handle in processes:
        handle.close()
    if any(code != 0 for code in return_codes):
        result = {"status": "validation_worker_error", "return_codes": return_codes}
        atomic_json(status_path, result)
        atomic_json(project_status, result | {"authoritative_plan": "IMPLEMENTATION_PLAN.md"})
        return result
    result = select(run)
    atomic_json(project_status, result | {"authoritative_plan": "IMPLEMENTATION_PLAN.md"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("watch", "evaluate", "select"))
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("device", nargs="?", default="cuda:1")
    args = parser.parse_args()
    if args.command == "watch":
        result = watch(args.run)
    elif args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires a checkpoint path")
        result = evaluate(args.run, args.checkpoint, args.device)
    else:
        result = select(args.run)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
