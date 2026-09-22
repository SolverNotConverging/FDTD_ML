#!/usr/bin/env python3
"""Run the stage-four multi-object pilot after the pair campaign qualifies."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from build_sparse_dataset_when_ready import campaign_ready

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-config", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--pair-output", type=Path, default=ROOT / "runs/sparse_pair_campaign_96")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_cluster_pilot_32.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_cluster_pilot_32")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not args.devices:
        raise ValueError("A positive poll interval and at least one device are required")
    pair_config, pair_output = args.pair_config.resolve(), args.pair_output.resolve()
    config, output = args.config.resolve(), args.output.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    while True:
        ready, message = campaign_ready(pair_config, pair_output)
        print(f"Pair campaign: {message}", flush=True)
        if ready or args.check_only:
            break
        time.sleep(args.poll_seconds)
    if args.check_only:
        return

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "scripts/run_sparse_pair_headroom_pilot.py"
    processes = []
    for shard, device in enumerate(args.devices):
        command = [
            sys.executable, "-u", str(runner), "--config", str(config),
            "--output", str(output), "--phase", "references", "--device", device,
            "--shard", str(shard), "--shards", str(len(args.devices)),
        ]
        log_path = logs / f"references_{shard}.log"
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log)
        processes.append(process)
        print(f"Cluster reference shard {shard}: pid={process.pid} device={device}", flush=True)
    exits = [process.wait() for process in processes]
    if any(exits):
        raise RuntimeError(f"Cluster reference shards failed: {exits}")
    print("Cluster reference shards complete; checking convergence before candidates", flush=True)
    subprocess.run(
        [
            sys.executable, "-u", str(ROOT / "scripts/launch_sparse_pair_candidates_when_ready.py"),
            "--config", str(config), "--output", str(output),
            "--devices", *args.devices, "--poll-seconds", str(args.poll_seconds),
        ],
        cwd=ROOT, env=environment, check=True,
    )
    print(f"Cluster pilot report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
