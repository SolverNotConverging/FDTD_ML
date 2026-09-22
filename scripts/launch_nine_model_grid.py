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


def initial_checkpoints(grid):
    """Verify all nine completed checkpoints before a matched warm start."""
    import torch

    grid = Path(grid).resolve()
    manifest = json.loads((grid / "launch.json").read_text())
    completion = json.loads((grid / "completion.json").read_text())
    entries = manifest["entries"]
    expected = {f"r{resolution}_c{channels}" for resolution in (128, 256, 384)
                for channels in (16, 24, 32)}
    if {entry["name"] for entry in entries} != expected or len(entries) != 9:
        raise ValueError("Initial grid does not contain exactly the nine expected models")
    if set(completion["exit_codes"]) != expected or any(completion["exit_codes"].values()):
        raise ValueError("Initial grid did not finish all nine fits successfully")
    checkpoints = {}
    for entry in entries:
        name = entry["name"]
        output = Path(entry["output"])
        summary = json.loads((output / "summary.json").read_text())
        checkpoint = output / "checkpoint.pt"
        if summary.get("status") != "complete" or summary.get("checkpoint_sha256") != _sha256_file(checkpoint):
            raise ValueError(f"Initial checkpoint is incomplete or changed: {name}")
        contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (contents.get("config", {}).get("raster_resolution") != entry["resolution"]
                or contents.get("model_kwargs", {}).get("base_channels") != entry["base_channels"]
                or contents.get("config", {}).get("input_schema") != "sparse_v2"):
            raise ValueError(f"Initial checkpoint identity differs from launch entry: {name}")
        checkpoints[name] = checkpoint.resolve()
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--init-grid", type=Path)
    parser.add_argument("--finetune-learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    metadata = json.loads(dataset.read_text())
    if metadata.get("schema_version") != 2:
        raise ValueError("Nine-model grid requires the merged sparse-v2 dataset")
    checkpoints = initial_checkpoints(args.init_grid) if args.init_grid else {}
    if checkpoints and args.finetune_learning_rate <= 0:
        raise ValueError("Fine-tuning learning rate must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    processes = []
    entries = []
    for index, config in enumerate(grid_configs()):
        resolution = config["raster_resolution"]
        channels = config["base_channels"]
        name = f"r{resolution}_c{channels}"
        if checkpoints:
            config.update({
                "seed": 20260924,
                "epochs": 120,
                "patience": 24,
                "learning_rate": args.finetune_learning_rate,
            })
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
        if checkpoints:
            command.extend(("--init-checkpoint", str(checkpoints[name])))
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
                "initial_checkpoint": str(checkpoints[name]) if checkpoints else None,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": _sha256_file(dataset),
        "dataset_arrays_sha256": _sha256_file(dataset.parent / metadata["arrays"]),
        "initial_grid": str(args.init_grid.resolve()) if args.init_grid else None,
        "initial_checkpoint_sha256": {
            name: _sha256_file(checkpoint) for name, checkpoint in checkpoints.items()
        },
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
