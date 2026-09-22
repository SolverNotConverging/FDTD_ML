#!/usr/bin/env python3
"""Create fixed 3x3 configs and run nine sparse-input training fits."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def grid_configs():
    for resolution, microbatch in ((128, 32), (256, 12), (384, 6)):
        for channels in (16, 24, 32):
            yield {
                "schema_version": 2,
                "seed": 20260922,
                "epochs": 180,
                "patience": 36,
                "report_every": 5,
                "batch_size": 96,
                "micro_batch_size": microbatch,
                "raster_resolution": resolution,
                "input_schema": "sparse_v2",
                "family_sampling_weights": {"sparse": 0.70, "simple": 0.30},
                "base_channels": channels,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "max_grading_ratio": 3.0,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    metadata = json.loads(dataset.read_text())
    if metadata.get("schema_version") != 2:
        raise ValueError("Nine-model grid requires the merged sparse-v2 dataset")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    processes = []
    entries = []
    for index, config in enumerate(grid_configs()):
        resolution = config["raster_resolution"]
        channels = config["base_channels"]
        name = f"r{resolution}_c{channels}"
        config_path = output / "configs" / f"{name}.json"
        run_output = output / name
        log_path = output / "logs" / f"{name}.log"
        _atomic_json(config_path, config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        device = args.devices[index % len(args.devices)]
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/train_mesh_distillation.py"),
            "--dataset",
            str(dataset),
            "--config",
            str(config_path),
            "--output",
            str(run_output),
            "--device",
            device,
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["OPENBLAS_NUM_THREADS"] = "1"
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
        processes.append(process)
        entries.append(
            {
                "name": name,
                "resolution": resolution,
                "base_channels": channels,
                "device": device,
                "pid": process.pid,
                "config": str(config_path),
                "output": str(run_output),
                "log": str(log_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": _sha256_file(dataset),
        "dataset_arrays_sha256": _sha256_file(dataset.parent / metadata["arrays"]),
        "entries": entries,
    }
    _atomic_json(output / "launch.json", manifest)
    print(json.dumps({"launch": str(output / "launch.json"), "jobs": entries}, indent=2), flush=True)
    exits = [process.wait() for process in processes]
    result = {**manifest, "exit_codes": dict(zip((entry["name"] for entry in entries), exits))}
    _atomic_json(output / "completion.json", result)
    print(json.dumps({"exit_codes": result["exit_codes"]}, indent=2), flush=True)
    if any(exits):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
