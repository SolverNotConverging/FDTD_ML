#!/usr/bin/env python3
"""Wait for qualified fine references, then run the sparse candidate campaign."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import run_sparse_pair_headroom_pilot as campaign

from scattermesh.metrics import scattering_loss

ROOT = Path(__file__).resolve().parents[1]


def references_ready(config, output, sources):
    try:
        records = campaign._load_records(config, output, "references", sources)
    except ValueError as error:
        return False, str(error)
    unsettled = [row["case_id"] for row in records if not row["accepted"]]
    if unsettled:
        raise ValueError(f"Reference did not settle: {unsettled[:5]}")
    by_key = {
        (row["definition"]["scene"]["scene_id"], row["definition"]["cells"]): row
        for row in records
    }
    coarse_cells, fine_cells = config["reference_budgets"][-2:]
    losses = []
    for scene in config["scenes"]:
        scene_id = scene["scene_id"]
        fields = []
        for cells in (coarse_cells, fine_cells):
            definition = by_key[(scene_id, cells)]["definition"]
            _, arrays_path = campaign._case_paths(output, definition)
            with np.load(arrays_path) as arrays:
                fields.append(arrays["complex_far_field"].copy())
        loss = scattering_loss(fields[1], fields[0])["joint_scattering_loss"]
        losses.append((loss, scene_id))
    worst = max(losses)
    limit = config["gate"]["maximum_reference_convergence_loss"]
    if worst[0] > limit:
        raise ValueError(f"Reference convergence failed: {worst}, limit={limit}")
    return True, f"{len(records)} accepted; worst 192/256 joint loss={worst[0]:.6g} ({worst[1]})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_pair_campaign_96")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not args.devices:
        raise ValueError("A positive poll interval and at least one device are required")
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = campaign.load_config(config_path)
    sources = campaign.source_hashes(config_path)
    while True:
        ready, message = references_ready(config, output, sources)
        print(f"References: {message}", flush=True)
        if ready or args.check_only:
            break
        time.sleep(args.poll_seconds)
    if args.check_only:
        return
    if campaign.source_hashes(config_path) != sources:
        raise ValueError("Numerical source or campaign config changed while references ran")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    logs = output / "candidate_logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard, device in enumerate(args.devices):
        command = [
            sys.executable, "-u", str(ROOT / "scripts/run_sparse_pair_headroom_pilot.py"),
            "--config", str(config_path), "--output", str(output),
            "--phase", "candidates", "--device", device,
            "--shard", str(shard), "--shards", str(len(args.devices)),
        ]
        log_path = logs / f"shard_{shard}.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
        processes.append(process)
        print(f"Candidate shard {shard}: pid={process.pid} device={device} log={log_path}", flush=True)
    exits = [process.wait() for process in processes]
    print(f"Candidate exit codes: {exits}", flush=True)
    if any(exits):
        raise SystemExit(1)
    if campaign.source_hashes(config_path) != sources:
        raise ValueError("Numerical source or campaign config changed during candidate runs")
    report = campaign.summarize(config, output, sources)
    print(json.dumps({"decision": report["decision"], "checks": report["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
