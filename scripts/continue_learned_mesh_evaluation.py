#!/usr/bin/env python3
"""Wait for full training, then run and summarize held-out physics shards."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_snapshot(dataset, training_output):
    dataset_path = Path(dataset)
    summary_path = Path(training_output) / "summary.json"
    checkpoint_path = Path(training_output) / "checkpoint.pt"
    snapshot = {
        "dataset_present": dataset_path.exists(),
        "summary_present": summary_path.exists(),
        "checkpoint_present": checkpoint_path.exists(),
        "ready": False,
    }
    if not summary_path.exists():
        return snapshot
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        snapshot["summary_malformed"] = True
        return snapshot
    snapshot["training_status"] = summary.get("status")
    snapshot["epochs_completed"] = summary.get("epochs_completed")
    snapshot["best_epoch"] = summary.get("best_epoch")
    provenance_matches = False
    if dataset_path.exists() and checkpoint_path.exists():
        try:
            metadata = json.loads(dataset_path.read_text())
            arrays_path = dataset_path.parent / metadata["arrays"]
            sources = summary.get("source_hashes", {})
            provenance_matches = bool(
                sources.get("dataset") == sha256_file(dataset_path)
                and sources.get("dataset_arrays") == sha256_file(arrays_path)
                and summary.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            provenance_matches = False
    snapshot["provenance_matches"] = provenance_matches
    snapshot["ready"] = bool(
        dataset_path.exists()
        and checkpoint_path.exists()
        and summary.get("status") == "complete"
        and provenance_matches
    )
    return snapshot


def expected_case_ids(dataset):
    payload = json.loads(Path(dataset).read_text())
    return [
        f"{example['sample_id']}_cnn"
        for example in payload["examples"]
        if example["split"] in {"validation", "test"}
    ]


def evaluation_snapshot(case_ids, output):
    statuses = {}
    malformed = 0
    for case_id in case_ids:
        path = Path(output) / "cases" / case_id / "record.json"
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text())
            statuses[case_id] = str(record["status"])
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            malformed += 1
    accepted = sum(status == "accepted" for status in statuses.values())
    unsettled = sum(status != "accepted" for status in statuses.values())
    return {
        "planned": len(case_ids),
        "completed": len(statuses),
        "accepted": accepted,
        "unsettled": unsettled,
        "malformed": malformed,
        "remaining": len(case_ids) - len(statuses),
    }


def worker_command(args, runner, shard, device):
    return [
        sys.executable,
        runner,
        "--dataset",
        args.dataset,
        "--checkpoint",
        args.training_output / "checkpoint.pt",
        "--candidate-output",
        args.candidate_output,
        "--output",
        args.output,
        "--device",
        device,
        "--shard",
        str(shard),
        "--shards",
        str(len(args.devices)),
        "--shard-mode",
        "hash",
        "--max-ratio",
        str(args.max_ratio),
    ]


def run_workers(args, runner, workflow_path, case_ids, environment):
    pending = set(range(len(args.devices)))
    attempts = {shard: 0 for shard in pending}
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending:
        processes = {}
        handles = {}
        for shard in sorted(pending):
            attempts[shard] += 1
            log_path = logs / f"shard_{shard}_attempt_{attempts[shard]}.log"
            handle = log_path.open("w")
            handles[shard] = handle
            command = worker_command(args, runner, shard, args.devices[shard])
            processes[shard] = subprocess.Popen(
                [str(value) for value in command],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        while any(process.poll() is None for process in processes.values()):
            atomic_json(
                workflow_path,
                {
                    "schema_version": 1,
                    "stage": "physics_evaluation",
                    "active_shards": [
                        shard for shard, process in processes.items() if process.poll() is None
                    ],
                    "shard_attempts": attempts,
                    **evaluation_snapshot(case_ids, args.output),
                },
            )
            time.sleep(args.poll_seconds)
        failed = set()
        for shard, process in processes.items():
            handles[shard].close()
            if process.returncode:
                failed.add(shard)
        exhausted = [shard for shard in failed if attempts[shard] > args.worker_retries]
        if exhausted:
            snapshot = evaluation_snapshot(case_ids, args.output)
            atomic_json(
                workflow_path,
                {
                    "schema_version": 1,
                    "stage": "worker_failed",
                    "failed_shards": sorted(exhausted),
                    "shard_attempts": attempts,
                    **snapshot,
                },
            )
            raise RuntimeError(f"Physics shards exhausted retry limit: {sorted(exhausted)}")
        pending = failed


def run_summary(args, runner, environment):
    command = [
        sys.executable,
        runner,
        "--dataset",
        args.dataset,
        "--checkpoint",
        args.training_output / "checkpoint.pt",
        "--candidate-output",
        args.candidate_output,
        "--output",
        args.output,
        "--summarize",
        "--success-decision",
        "passes_frozen_physics_evaluation",
        "--max-ratio",
        str(args.max_ratio),
    ]
    subprocess.run([str(value) for value in command], check=True, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--max-ratio", type=float, default=3.0)
    parser.add_argument("--worker-retries", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if not args.devices:
        raise ValueError("At least one device is required")
    if args.worker_retries < 0:
        raise ValueError("worker-retries must be nonnegative")
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be in [1, 60]")

    workflow_path = args.output / "workflow.json"
    last_snapshot = None
    while True:
        snapshot = training_snapshot(args.dataset, args.training_output)
        if snapshot != last_snapshot:
            print(json.dumps(snapshot), flush=True)
            last_snapshot = snapshot
        atomic_json(
            workflow_path,
            {"schema_version": 1, "stage": "waiting_for_training", **snapshot},
        )
        if snapshot["ready"]:
            break
        if args.no_wait:
            raise SystemExit(2)
        time.sleep(args.poll_seconds)

    case_ids = expected_case_ids(args.dataset)
    if not case_ids:
        raise ValueError("Dataset has no validation/test examples")
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/scattermesh-mpl",
    }
    runner = Path(__file__).with_name("run_learned_mesh_pilot.py")
    atomic_json(
        workflow_path,
        {
            "schema_version": 1,
            "stage": "physics_evaluation",
            "source_hashes": {
                "dataset": sha256_file(args.dataset),
                "checkpoint": sha256_file(args.training_output / "checkpoint.pt"),
                "runner": sha256_file(runner),
            },
            **evaluation_snapshot(case_ids, args.output),
        },
    )
    run_workers(args, runner, workflow_path, case_ids, environment)
    atomic_json(
        workflow_path,
        {"schema_version": 1, "stage": "summarizing", **evaluation_snapshot(case_ids, args.output)},
    )
    run_summary(args, runner, environment)
    report = json.loads((args.output / "report.json").read_text())
    stage = (
        "complete"
        if report.get("decision") == "passes_frozen_physics_evaluation"
        else "physics_gate_failed"
    )
    atomic_json(
        workflow_path,
        {"schema_version": 1, "stage": stage, "report": report},
    )
    if stage != "complete":
        raise RuntimeError("Full learned-mesh physics evaluation did not pass")


if __name__ == "__main__":
    main()
