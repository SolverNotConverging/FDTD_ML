#!/usr/bin/env python3
"""Wait for the main frozen gate, then run the circle OOD evaluation."""

import argparse
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


def prerequisite_snapshot(plan, checkpoint, main_physics_output):
    report_path = Path(main_physics_output) / "report.json"
    snapshot = {
        "plan_present": Path(plan).exists(),
        "checkpoint_present": Path(checkpoint).exists(),
        "main_physics_report_present": report_path.exists(),
        "ready": False,
    }
    if not report_path.exists():
        return snapshot
    try:
        report = json.loads(report_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        snapshot["main_physics_report_malformed"] = True
        return snapshot
    snapshot["main_physics_decision"] = report.get("decision")
    snapshot["ready"] = bool(
        snapshot["plan_present"]
        and snapshot["checkpoint_present"]
        and report.get("decision") == "passes_frozen_physics_evaluation"
    )
    return snapshot


def expected_case_ids(plan):
    payload = json.loads(Path(plan).read_text())
    return [
        f"{example['sample_id']}_{mesh_kind}"
        for example in payload["examples"]
        for mesh_kind in ("uniform", "cnn")
    ]


def evaluation_snapshot(case_ids, output):
    statuses = {}
    malformed = 0
    for case_id in case_ids:
        path = Path(output) / "cases" / case_id / "record.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            statuses[case_id] = str(payload["status"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            malformed += 1
    return {
        "planned": len(case_ids),
        "completed": len(statuses),
        "accepted": sum(status == "accepted" for status in statuses.values()),
        "unsettled": sum(status != "accepted" for status in statuses.values()),
        "malformed": malformed,
        "remaining": len(case_ids) - len(statuses),
    }


def worker_command(args, runner, shard, device):
    return [
        sys.executable,
        runner,
        "--plan",
        args.plan,
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--device",
        device,
        "--shard",
        str(shard),
        "--shards",
        str(len(args.devices)),
    ]


def run_workers(args, runner, workflow_path, case_ids, environment):
    pending = set(range(len(args.devices)))
    attempts = {shard: 0 for shard in pending}
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending:
        processes, handles = {}, {}
        for shard in sorted(pending):
            attempts[shard] += 1
            log_path = logs / f"shard_{shard}_attempt_{attempts[shard]}.log"
            handles[shard] = log_path.open("w")
            processes[shard] = subprocess.Popen(
                [str(value) for value in worker_command(args, runner, shard, args.devices[shard])],
                stdout=handles[shard],
                stderr=subprocess.STDOUT,
                env=environment,
            )
        while any(process.poll() is None for process in processes.values()):
            atomic_json(
                workflow_path,
                {
                    "schema_version": 1,
                    "stage": "circle_generalization_evaluation",
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
            atomic_json(
                workflow_path,
                {
                    "schema_version": 1,
                    "stage": "worker_failed",
                    "failed_shards": sorted(exhausted),
                    "shard_attempts": attempts,
                    **evaluation_snapshot(case_ids, args.output),
                },
            )
            raise RuntimeError(f"Generalization shards exhausted retries: {sorted(exhausted)}")
        pending = failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/circle_position_scale_generalization.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/simple_factorial_exact_training_6916879/checkpoint.pt"),
    )
    parser.add_argument(
        "--main-physics-output",
        type=Path,
        default=Path("runs/simple_factorial_exact_physics_6916879"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/circle_position_scale_generalization_6916879"),
    )
    parser.add_argument(
        "--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
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
    previous = None
    while True:
        snapshot = prerequisite_snapshot(args.plan, args.checkpoint, args.main_physics_output)
        if snapshot != previous:
            print(json.dumps(snapshot), flush=True)
            previous = snapshot
        atomic_json(
            workflow_path,
            {"schema_version": 1, "stage": "waiting_for_main_physics", **snapshot},
        )
        if snapshot["ready"]:
            break
        if args.no_wait:
            raise SystemExit(2)
        time.sleep(args.poll_seconds)

    case_ids = expected_case_ids(args.plan)
    runner = Path(__file__).with_name("run_circle_generalization_evaluation.py")
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/scattermesh-mpl",
    }
    run_workers(args, runner, workflow_path, case_ids, environment)
    atomic_json(
        workflow_path,
        {"schema_version": 1, "stage": "summarizing", **evaluation_snapshot(case_ids, args.output)},
    )
    subprocess.run(
        [
            sys.executable,
            runner,
            "--plan",
            args.plan,
            "--checkpoint",
            args.checkpoint,
            "--output",
            args.output,
            "--summarize",
        ],
        check=True,
        env=environment,
    )
    report = json.loads((args.output / "report.json").read_text())
    passed = report.get("decision") == "passes_circle_position_scale_generalization"
    atomic_json(
        workflow_path,
        {
            "schema_version": 1,
            "stage": "complete" if passed else "generalization_gate_failed",
            "report": report,
        },
    )
    if not passed:
        raise RuntimeError("Circle position/scale generalization gate failed")


if __name__ == "__main__":
    main()
