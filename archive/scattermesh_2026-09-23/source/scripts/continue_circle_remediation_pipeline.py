#!/usr/bin/env python3
"""Continue circle remediation through retraining and both frozen physics gates."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def campaign_snapshot(campaign_path, output):
    campaign = json.loads(Path(campaign_path).read_text())
    statuses = Counter()
    malformed = 0
    for condition_id in campaign["condition_ids"]:
        for candidate in campaign["candidate_names"]:
            path = Path(output) / "cases" / f"{condition_id}_{candidate}" / "record.json"
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
                attempt = record.get("config", {}).get("duration_attempt_index")
                schedule = campaign.get("duration_schedule_s")
                if (
                    record.get("status") == "unsettled"
                    and schedule is not None
                    and attempt is not None
                    and int(attempt) < len(schedule) - 1
                ):
                    continue
                statuses[str(record["status"])] += 1
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                malformed += 1
    planned = len(campaign["condition_ids"]) * len(campaign["candidate_names"])
    completed = sum(statuses.values())
    return {
        "planned": planned,
        "completed": completed,
        "remaining": planned - completed,
        "malformed": malformed,
        "status_counts": dict(sorted(statuses.items())),
        "ready": completed == planned and malformed == 0,
    }


def run(command, environment):
    command = [str(value) for value in command]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--augmentation-output", type=Path, required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--main-physics-output", type=Path, required=True)
    parser.add_argument("--circle-plan", type=Path, required=True)
    parser.add_argument("--circle-output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if not args.devices or not 1 <= args.poll_seconds <= 60:
        raise ValueError("Devices are required and poll-seconds must be in [1, 60]")

    workflow = args.training_output / "remediation_workflow.json"
    previous = None
    while True:
        snapshot = campaign_snapshot(args.campaign, args.campaign_output)
        if snapshot != previous:
            print(json.dumps(snapshot), flush=True)
            previous = snapshot
        atomic_json(workflow, {"schema_version": 1, "stage": "waiting_for_labels", **snapshot})
        if snapshot["ready"]:
            break
        if args.no_wait:
            raise SystemExit(2)
        time.sleep(args.poll_seconds)

    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/scattermesh-mpl",
    }
    scripts = Path(__file__).resolve().parent
    atomic_json(workflow, {"schema_version": 1, "stage": "summarizing_labels"})
    run(
        [
            sys.executable,
            scripts / "run_simple_candidate_campaign.py",
            "--manifest",
            args.manifest,
            "--campaign",
            args.campaign,
            "--output",
            args.campaign_output,
            "--summarize",
        ],
        environment,
    )
    report = json.loads((args.campaign_output / "report.json").read_text())
    readiness = report.get("training_readiness", {})
    if (
        report.get("decision") != "accepted"
        or readiness.get("decision") != "ready_for_m5_pilot"
        or not all(readiness.get("checks", {}).values())
    ):
        atomic_json(
            workflow,
            {"schema_version": 1, "stage": "label_gate_failed", "report": report},
        )
        raise RuntimeError("Circle remediation labels did not pass the training gate")

    atomic_json(workflow, {"schema_version": 1, "stage": "building_augmentation"})
    run(
        [
            sys.executable,
            scripts / "build_mesh_distillation_dataset.py",
            "--manifest",
            args.manifest,
            "--campaign",
            args.campaign,
            "--campaign-output",
            args.campaign_output,
            "--output",
            args.augmentation_output,
        ],
        environment,
    )
    atomic_json(workflow, {"schema_version": 1, "stage": "merging_training_rows"})
    run(
        [
            sys.executable,
            scripts / "merge_mesh_distillation_datasets.py",
            "--base",
            args.base_dataset,
            "--augmentation",
            args.augmentation_output / "dataset.json",
            "--output",
            args.merged_output,
        ],
        environment,
    )
    merged_dataset = args.merged_output / "dataset.json"
    atomic_json(workflow, {"schema_version": 1, "stage": "training"})
    run(
        [
            sys.executable,
            scripts / "train_mesh_distillation.py",
            "--dataset",
            merged_dataset,
            "--config",
            args.training_config,
            "--output",
            args.training_output,
            "--device",
            args.devices[0],
        ],
        environment,
    )

    atomic_json(workflow, {"schema_version": 1, "stage": "main_frozen_physics"})
    run(
        [
            sys.executable,
            scripts / "continue_learned_mesh_evaluation.py",
            "--dataset",
            merged_dataset,
            "--training-output",
            args.training_output,
            "--candidate-output",
            args.candidate_output,
            "--output",
            args.main_physics_output,
            "--devices",
            *args.devices,
        ],
        environment,
    )
    main_report = json.loads((args.main_physics_output / "report.json").read_text())
    if main_report.get("decision") != "passes_frozen_physics_evaluation":
        atomic_json(
            workflow,
            {"schema_version": 1, "stage": "main_physics_gate_failed", "report": main_report},
        )
        raise RuntimeError("Retrained model failed the main frozen physics gate")

    atomic_json(workflow, {"schema_version": 1, "stage": "circle_frozen_physics"})
    run(
        [
            sys.executable,
            scripts / "continue_circle_generalization_evaluation.py",
            "--plan",
            args.circle_plan,
            "--checkpoint",
            args.training_output / "checkpoint.pt",
            "--main-physics-output",
            args.main_physics_output,
            "--output",
            args.circle_output,
            "--devices",
            *args.devices,
        ],
        environment,
    )
    circle_report = json.loads((args.circle_output / "report.json").read_text())
    atomic_json(
        workflow,
        {
            "schema_version": 1,
            "stage": "complete" if circle_report.get("decision") == "passes_circle_position_scale_generalization" else "circle_gate_failed",
            "main_physics_report": str(args.main_physics_output / "report.json"),
            "circle_report": circle_report,
        },
    )
    if circle_report.get("decision") != "passes_circle_position_scale_generalization":
        raise RuntimeError("Retrained model failed the circle position/scale gate")


if __name__ == "__main__":
    main()
