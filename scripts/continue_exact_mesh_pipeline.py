#!/usr/bin/env python3
"""Wait for qualified labels, then build the dataset and train the full model."""

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


def campaign_snapshot(campaign, campaign_output):
    config = json.loads(Path(campaign).read_text())
    duration_schedule = config.get("duration_schedule_s")
    expected = [
        f"{condition}_{candidate}"
        for condition in config["condition_ids"]
        for candidate in config["candidate_names"]
    ]
    statuses = Counter()
    retrying = Counter()
    malformed = 0
    for case_id in expected:
        path = Path(campaign_output) / "cases" / case_id / "record.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            status = str(payload["status"])
            attempt_index = payload.get("config", {}).get("duration_attempt_index")
            retryable = (
                status == "unsettled"
                and duration_schedule is not None
                and attempt_index is not None
                and int(attempt_index) < len(duration_schedule) - 1
            )
            if retryable:
                retrying[status] += 1
            else:
                statuses[status] += 1
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            malformed += 1
    completed = sum(statuses.values())
    return {
        "planned": len(expected),
        "completed": completed,
        "remaining": len(expected) - completed,
        "malformed": malformed,
        "status_counts": dict(sorted(statuses.items())),
        "retrying_counts": dict(sorted(retrying.items())),
        "ready": completed == len(expected)
        and malformed == 0,
    }


def run(command, environment):
    print("running:", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be in [1, 60]")
    workflow_path = args.training_output / "workflow.json"
    last_completed = None
    while True:
        snapshot = campaign_snapshot(args.campaign, args.campaign_output)
        if snapshot["completed"] != last_completed:
            print(json.dumps(snapshot), flush=True)
            last_completed = snapshot["completed"]
        atomic_json(
            workflow_path,
            {"schema_version": 1, "stage": "waiting_for_terminal_labels", **snapshot},
        )
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
    runner = Path(__file__).with_name("run_simple_candidate_campaign.py")
    builder = Path(__file__).with_name("build_mesh_distillation_dataset.py")
    trainer = Path(__file__).with_name("train_mesh_distillation.py")
    atomic_json(workflow_path, {"schema_version": 1, "stage": "summarizing_labels"})
    run(
        [
            sys.executable,
            runner,
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
            workflow_path,
            {"schema_version": 1, "stage": "label_gate_failed", "report": report},
        )
        raise RuntimeError("Full campaign did not pass its label gate")

    atomic_json(workflow_path, {"schema_version": 1, "stage": "building_dataset"})
    run(
        [
            sys.executable,
            builder,
            "--manifest",
            args.manifest,
            "--campaign",
            args.campaign,
            "--campaign-output",
            args.campaign_output,
            "--output",
            args.dataset_output,
        ],
        environment,
    )
    atomic_json(workflow_path, {"schema_version": 1, "stage": "training"})
    run(
        [
            sys.executable,
            trainer,
            "--dataset",
            args.dataset_output / "dataset.json",
            "--config",
            args.training_config,
            "--output",
            args.training_output,
            "--device",
            args.device,
        ],
        environment,
    )
    summary = json.loads((args.training_output / "summary.json").read_text())
    atomic_json(
        workflow_path,
        {"schema_version": 1, "stage": "complete", "training_summary": summary},
    )


if __name__ == "__main__":
    main()
