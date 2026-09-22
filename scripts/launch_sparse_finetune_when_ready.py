#!/usr/bin/env python3
"""Warm-start all nine CNNs after the expanded sparse-pair dataset is verified."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_ready(dataset, base, config, campaign):
    if not dataset.is_file():
        return False, "waiting for expanded dataset"
    metadata = json.loads(dataset.read_text())
    arrays = dataset.parent / metadata["arrays"]
    if not arrays.is_file():
        return False, "waiting for expanded dataset arrays"
    base_metadata = json.loads(base.read_text())
    campaign_config = json.loads(config.read_text())
    expected = len(base_metadata["examples"]) + len(campaign_config["scenes"]) * len(
        campaign_config["budgets"]
    )
    if metadata.get("schema_version") != 2 or metadata.get("example_count") != expected:
        raise ValueError("Expanded dataset does not cover all declared conditions")
    if len(metadata["examples"]) != expected or sum(metadata["split_counts"].values()) != expected:
        raise ValueError("Expanded dataset example/split counts are inconsistent")
    base_arrays = base.parent / base_metadata["arrays"]
    report = campaign / "report.json"
    if not report.is_file():
        raise ValueError("Expanded dataset exists without a campaign report")
    result = json.loads(report.read_text())
    checks = result.get("checks")
    if result.get("decision") != "passes_sparse_pair_headroom" or not checks or not all(
        checks.values()
    ):
        raise ValueError("Expanded dataset campaign no longer passes its frozen gate")
    expected_hashes = {
        "base_dataset": _sha256_file(base),
        "base_arrays": _sha256_file(base_arrays),
        "pilot_0_config": _sha256_file(config),
        "pilot_0_report": _sha256_file(report),
    }
    if metadata.get("source_hashes") != expected_hashes:
        raise ValueError("Expanded dataset provenance differs from current campaign")
    return True, f"{expected} examples verified; arrays sha256={_sha256_file(arrays)}"


def initial_grid_ready(grid):
    launch_path = grid / "launch.json"
    completion_path = grid / "completion.json"
    if not launch_path.is_file() or not completion_path.is_file():
        return False, "waiting for initial nine-model grid"
    launch = json.loads(launch_path.read_text())
    exits = json.loads(completion_path.read_text())["exit_codes"]
    expected = {f"r{resolution}_c{channels}" for resolution in (128, 256, 384)
                for channels in (16, 24, 32)}
    entries = launch["entries"]
    if len(entries) != 9 or {entry["name"] for entry in entries} != expected:
        raise ValueError("Initial grid does not contain the expected nine models")
    if set(exits) != expected or any(exits.values()):
        raise ValueError(f"Initial grid has failed or missing fits: {exits}")
    for entry in entries:
        output = Path(entry["output"])
        summary_path, checkpoint_path = output / "summary.json", output / "checkpoint.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            return False, f"waiting for initial checkpoint {entry['name']}"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise ValueError(f"Initial checkpoint is incomplete or changed: {entry['name']}")
    return True, "all nine initial checkpoints verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "runs/sparse_joint_dataset_96/dataset.json")
    parser.add_argument("--base", type=Path, default=ROOT / "runs/sparse_joint_dataset_pairs/dataset.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/sparse_pair_campaign_96")
    parser.add_argument("--init-grid", type=Path, default=ROOT / "runs/sparse_nine_model_grid")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_nine_model_finetune_96")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--family-weight", action="append", metavar="NAME=FRACTION")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not args.devices:
        raise ValueError("A positive poll interval and at least one device are required")
    dataset, base = args.dataset.resolve(), args.base.resolve()
    config, campaign = args.config.resolve(), args.campaign.resolve()
    while True:
        ready, message = dataset_ready(dataset, base, config, campaign)
        if ready:
            ready, message = initial_grid_ready(args.init_grid.resolve())
        print(message, flush=True)
        if ready or args.check_only:
            break
        time.sleep(args.poll_seconds)
    if args.check_only:
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    command = [
        sys.executable, "-u", str(ROOT / "scripts/launch_nine_model_grid.py"),
        "--dataset", str(dataset), "--output", str(args.output.resolve()),
        "--init-grid", str(args.init_grid.resolve()), "--devices", *args.devices,
    ]
    for weight in args.family_weight or []:
        command.extend(("--family-weight", weight))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
