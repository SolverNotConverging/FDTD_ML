#!/usr/bin/env python3
"""Run a restartable dielectric-reference plan across explicitly assigned GPUs."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "scripts/run_dielectric_reference_attempt.py"
TERMINAL = {"succeeded", "nonconverged", "failed"}
TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
DEVICE = re.compile(r"^cuda:[0-9]+$")
SCENES = {"eps12_small", "eps30_small", "eps30_lossy"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def plan_sha256(plan):
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_sha256():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/scattermesh").glob("*.py"))
    }


def validate_plan(raw, plan_path, devices):
    if raw.get("schema_version") != 1 or not isinstance(raw.get("campaign_id"), str):
        raise ValueError("Plan requires schema_version 1 and a campaign_id")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Plan requires a nonempty tasks list")
    output_root = Path(raw.get("output_root", "runs/dielectric_reference_campaign"))
    if not output_root.is_absolute():
        output_root = (plan_path.parent / output_root).resolve()
    validated, identifiers = [], set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task {index} must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id):
            raise ValueError(f"Task {index} has an invalid task_id")
        if task_id in identifiers:
            raise ValueError(f"Duplicate task_id: {task_id}")
        identifiers.add(task_id)
        scene = task.get("scene")
        if scene not in SCENES:
            raise ValueError(f"Task {task_id} has an unsupported scene")
        cells = task.get("cells")
        duration = task.get("duration_ns")
        samples = task.get("samples", 24)
        pml = task.get("pml", 0.15)
        larger = task.get("larger_contour", False)
        if (
            isinstance(cells, bool)
            or not isinstance(cells, int)
            or cells < 16
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < 1
            or isinstance(pml, bool)
            or not isinstance(pml, (int, float))
            or pml <= 0
            or not isinstance(larger, bool)
        ):
            raise ValueError(f"Task {task_id} has invalid numerical controls")
        device = task.get("device", devices[index % len(devices)])
        if device not in devices:
            raise ValueError(f"Task {task_id} uses unavailable device {device}")
        validated.append(
            dict(
                task_id=task_id,
                scene=scene,
                cells=cells,
                duration_ns=float(duration),
                samples=samples,
                pml=float(pml),
                larger_contour=larger,
                device=device,
            )
        )
    return dict(
        schema_version=1,
        campaign_id=raw["campaign_id"],
        output_root=str(output_root),
        tasks=validated,
    )


def task_matches(record, task, sources):
    config = record.get("config", {})
    expected_contour = [0.23, 0.97, 0.23, 0.97] if task["larger_contour"] else [0.3, 0.9, 0.3, 0.9]
    return (
        config.get("scene") == task["scene"]
        and config.get("cells") == task["cells"]
        and config.get("duration_ns") == task["duration_ns"]
        and config.get("samples") == task["samples"]
        and config.get("pml") == task["pml"]
        and config.get("contour") == expected_contour
        and config.get("device") == task["device"]
        and config.get("source_sha256") == sources
    )


def valid_artifact(record_path, record):
    config = record.get("config")
    if not isinstance(config, dict) or record.get("status") not in {"pass", "fail"}:
        return False
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if record.get("fingerprint") != expected:
        return False
    try:
        with np.load(record_path.with_name("spectra.npz")) as arrays:
            numerical = arrays["complex_numerical"]
            analytic = arrays["complex_analytic"]
            frequencies = arrays["frequencies"]
            angles = arrays["angles"]
            return bool(
                numerical.shape == (3, 180)
                and analytic.shape == numerical.shape
                and np.iscomplexobj(numerical)
                and np.iscomplexobj(analytic)
                and np.isfinite(numerical).all()
                and np.isfinite(analytic).all()
                and np.array_equal(frequencies, [0.8e9, 1.0e9, 1.2e9])
                and np.array_equal(angles, np.linspace(0, 2 * np.pi, 180, endpoint=False))
            )
    except (OSError, ValueError, KeyError):
        return False


def find_result(output_root, task, sources=None):
    sources = source_sha256() if sources is None else sources
    matches = []
    for path in output_root.glob(f"{task['scene']}/*/record.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if task_matches(record, task, sources) and valid_artifact(path, record):
            matches.append((path, record))
    if len(matches) > 1:
        raise ValueError(f"Multiple matching results for task {task['task_id']}")
    return matches[0] if matches else None


def build_command(task, output_root, runner=DEFAULT_RUNNER):
    command = [
        sys.executable,
        "-u",
        str(runner),
        "--scene",
        task["scene"],
        "--cells",
        str(task["cells"]),
        "--duration-ns",
        f"{task['duration_ns']:g}",
        "--device",
        task["device"],
        "--samples",
        str(task["samples"]),
        "--pml",
        f"{task['pml']:g}",
        "--output-root",
        str(output_root),
    ]
    if task["larger_contour"]:
        command.append("--larger-contour")
    return command


def initial_state(plan, digest):
    return dict(
        schema_version=1,
        campaign_id=plan["campaign_id"],
        plan_sha256=digest,
        output_root=plan["output_root"],
        coordinator_pid=os.getpid(),
        started_unix=time.time(),
        updated_unix=time.time(),
        status="running",
        tasks={
            task["task_id"]: dict(
                status="pending",
                device=task["device"],
                attempts=0,
                process_failures=[],
            )
            for task in plan["tasks"]
        },
    )


def load_state(path, plan, digest):
    if not path.exists():
        return initial_state(plan, digest)
    state = json.loads(path.read_text())
    if state.get("plan_sha256") != digest or state.get("campaign_id") != plan["campaign_id"]:
        raise ValueError("Campaign state belongs to a different immutable plan")
    for task in plan["tasks"]:
        row = state["tasks"].get(task["task_id"])
        if row is None or row.get("device") != task["device"]:
            raise ValueError(f"State assignment mismatch for task {task['task_id']}")
        if row.get("status") == "running":
            row["status"] = "pending"
            row["interrupted_unix"] = time.time()
    state["coordinator_pid"] = os.getpid()
    state["status"] = "running"
    return state


def reconcile(state, plan, sources):
    output_root = Path(plan["output_root"])
    for task in plan["tasks"]:
        row = state["tasks"][task["task_id"]]
        if row["status"] == "failed":
            continue
        result = find_result(output_root, task, sources)
        if result is None:
            if row["status"] in {"succeeded", "nonconverged"}:
                row["status"] = "pending"
                row["invalidated_result_unix"] = time.time()
            continue
        path, record = result
        row.update(
            status="succeeded" if record.get("status") == "pass" else "nonconverged",
            scientific_status=record.get("status"),
            result_path=str(path),
            result_fingerprint=record.get("fingerprint"),
            completed_unix=time.time(),
        )


def status_summary(state):
    counts = Counter(row["status"] for row in state["tasks"].values())
    return dict(
        campaign_id=state["campaign_id"],
        status=state["status"],
        counts=dict(sorted(counts.items())),
        total=len(state["tasks"]),
        started_unix=state["started_unix"],
        updated_unix=state["updated_unix"],
        tasks=state["tasks"],
    )


def run_campaign(plan, campaign_directory, *, runner=DEFAULT_RUNNER, max_attempts=2):
    digest = plan_sha256(plan)
    state_path = campaign_directory / "state.json"
    logs = campaign_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sources = source_sha256()
    state = load_state(state_path, plan, digest)
    reconcile(state, plan, sources)
    atomic_json(state_path, state)
    active = {}
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        SCATTERMESH_SOURCE_SHA256=json.dumps(sources, sort_keys=True),
    )
    while True:
        for task in plan["tasks"]:
            row = state["tasks"][task["task_id"]]
            if row["status"] != "pending" or task["device"] in active:
                continue
            log_path = logs / f"{task['task_id']}.log"
            handle = log_path.open("a")
            handle.write(
                f"\n=== attempt {row['attempts'] + 1} device={task['device']} "
                f"time={time.time():.6f} ===\n"
            )
            handle.flush()
            process = subprocess.Popen(
                build_command(task, Path(plan["output_root"]), runner),
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            row.update(
                status="running",
                attempts=row["attempts"] + 1,
                pid=process.pid,
                launched_unix=time.time(),
                log_path=str(log_path),
            )
            active[task["device"]] = (task["task_id"], process, handle)
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        finished = []
        for device, (task_id, process, handle) in active.items():
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            task, row = tasks[task_id], state["tasks"][task_id]
            row["returncode"] = returncode
            if returncode == 0:
                result = find_result(Path(plan["output_root"]), task, sources)
                if result is not None:
                    path, record = result
                    row.update(
                        status=("succeeded" if record.get("status") == "pass" else "nonconverged"),
                        scientific_status=record.get("status"),
                        result_path=str(path),
                        result_fingerprint=record.get("fingerprint"),
                        completed_unix=time.time(),
                    )
                else:
                    returncode = -1
                    row["returncode"] = returncode
            if returncode != 0:
                row["process_failures"].append(
                    dict(returncode=returncode, failed_unix=time.time(), attempt=row["attempts"])
                )
                row["status"] = "pending" if row["attempts"] < max_attempts else "failed"
            finished.append(device)
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)
        for device in finished:
            del active[device]

        if not active and all(row["status"] in TERMINAL for row in state["tasks"].values()):
            break
        if not active and not any(row["status"] == "pending" for row in state["tasks"].values()):
            raise RuntimeError("Campaign has no runnable or terminal tasks")
        time.sleep(0.2)

    counts = Counter(row["status"] for row in state["tasks"].values())
    state["status"] = "complete" if counts.get("failed", 0) == 0 else "complete_with_failures"
    state["completed_unix"] = time.time()
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    atomic_json(campaign_directory / "summary.json", status_summary(state))
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        not args.devices
        or len(set(args.devices)) != len(args.devices)
        or any(not DEVICE.fullmatch(device) for device in args.devices)
    ):
        raise ValueError("Devices must be unique cuda:N identifiers")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    output = args.output.resolve()
    state_path = output / "state.json"
    if args.status:
        if not state_path.exists():
            raise FileNotFoundError(f"No campaign state at {state_path}")
        print(json.dumps(status_summary(json.loads(state_path.read_text())), indent=2))
        return
    plan_path = args.plan.resolve()
    plan = validate_plan(json.loads(plan_path.read_text()), plan_path, args.devices)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another coordinator holds the campaign lock") from error
        state = run_campaign(
            plan, output, runner=args.runner.resolve(), max_attempts=args.max_attempts
        )
    print(json.dumps(status_summary(state), indent=2))


if __name__ == "__main__":
    main()
