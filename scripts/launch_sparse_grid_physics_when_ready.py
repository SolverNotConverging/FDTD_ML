#!/usr/bin/env python3
"""Evaluate all nine trained sparse CNNs against held-out scattering physics."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOTS = (
    (ROOT / "configs/sparse_pair_headroom_pilot.json", ROOT / "runs/sparse_pair_headroom_pilot"),
    (
        ROOT / "configs/sparse_mixed_pair_headroom_pilot.json",
        ROOT / "runs/sparse_mixed_pair_headroom_pilot",
    ),
)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def completed_models(grid):
    launch = json.loads((grid / "launch.json").read_text())
    if len(launch["entries"]) != 9:
        raise ValueError("Expected exactly nine capacity/resolution variants")
    ready = []
    for entry in launch["entries"]:
        output = Path(entry["output"])
        summary_path = output / "summary.json"
        checkpoint_path = output / "checkpoint.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            return None, f"waiting for {entry['name']}"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete":
            return None, f"waiting for {entry['name']} complete status"
        if summary.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise ValueError(f"Checkpoint hash mismatch: {entry['name']}")
        ready.append((entry["name"], checkpoint_path, summary))
    completion_path = grid / "completion.json"
    if not completion_path.is_file():
        return None, "waiting for nine-model launcher completion"
    exits = json.loads(completion_path.read_text())["exit_codes"]
    if any(code != 0 for code in exits.values()):
        raise ValueError(f"Training launcher reported failures: {exits}")
    return ready, "all nine checkpoints verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "runs/sparse_nine_model_grid")
    parser.add_argument("--dataset", type=Path, default=ROOT / "runs/sparse_joint_dataset_pairs/dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_nine_model_physics")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not args.devices:
        raise ValueError("A positive poll interval and at least one device are required")
    grid, dataset, output = args.grid.resolve(), args.dataset.resolve(), args.output.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    launch = json.loads((grid / "launch.json").read_text())
    dataset_arrays = dataset.parent / json.loads(dataset.read_text())["arrays"]
    if (launch["dataset_sha256"] != _sha256_file(dataset)
            or launch["dataset_arrays_sha256"] != _sha256_file(dataset_arrays)):
        raise ValueError("Evaluation dataset differs from the nine-model training dataset")
    while True:
        ready, message = completed_models(grid)
        print(f"Training: {message}", flush=True)
        if ready is not None or args.check_only:
            break
        time.sleep(args.poll_seconds)
    if args.check_only:
        return

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    evaluator = ROOT / "scripts/evaluate_sparse_model_grid.py"
    reports = {}
    for name, checkpoint_path, summary in ready:
        model_output = output / name
        logs = model_output / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        base = [
            sys.executable, "-u", str(evaluator), "--dataset", str(dataset),
            "--checkpoint", str(checkpoint_path), "--output", str(model_output),
        ]
        for config_path, pilot_output in PILOTS:
            base.extend(("--pilot", str(config_path), str(pilot_output)))
        processes = []
        for shard, device in enumerate(args.devices):
            command = [*base, "--device", device, "--shard", str(shard), "--shards", str(len(args.devices))]
            log_path = logs / f"shard_{shard}.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
            processes.append(process)
            print(f"{name} shard {shard}: pid={process.pid} device={device}", flush=True)
        exits = [process.wait() for process in processes]
        if any(exits):
            raise RuntimeError(f"Physics evaluation failed for {name}: {exits}")
        subprocess.run(
            [*base, "--summarize"], cwd=ROOT, env=environment, check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads((model_output / "report.json").read_text())
        reports[name] = {
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "parameter_count": summary["parameter_count"],
            "raster_resolution": int(name.split("_")[0][1:]),
            "base_channels": int(name.split("_")[1][1:]),
            "split_reports": result["split_reports"],
        }
        _atomic_json(output / "comparison.json", {"schema_version": 1, "models": reports})
        validation = result["split_reports"]["validation"]
        print(f"{name} validation: {validation['accepted_count']}/{validation['case_count']} "
              f"accepted, median improvement={validation['median_improvement']}", flush=True)
    print(f"Physics comparison complete: {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
