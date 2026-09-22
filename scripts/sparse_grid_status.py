#!/usr/bin/env python3
"""Print compact progress for the nine CNN fits and sparse reference campaign."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("runs/sparse_nine_model_grid"))
    parser.add_argument("--references", type=Path, default=Path("runs/sparse_pair_campaign_96"))
    parser.add_argument("--scene-count", type=int, default=96)
    args = parser.parse_args()
    launch_path = args.grid / "launch.json"
    if not launch_path.is_file():
        raise SystemExit(f"Training launch is missing: {launch_path}")
    launch = json.loads(launch_path.read_text())
    print("Nine-model training")
    for entry in launch["entries"]:
        output = Path(entry["output"])
        progress_path = output / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            status = progress.get("status", "unknown")
            epoch = progress.get("epoch", progress.get("epochs_completed", "-"))
            best = progress.get("best_epoch", "-")
            best_loss = progress.get("best_validation_loss")
            loss = f"{best_loss:.6g}" if best_loss is not None else "-"
        else:
            status, epoch, best, loss = "starting", "-", "-", "-"
        log_path = Path(entry["log"])
        if status == "starting" and log_path.is_file():
            lines = log_path.read_text().splitlines()
            if lines and ("Traceback" in lines[-1] or "Error" in lines[-1]):
                status = "error; inspect log"
        print(f"  {entry['name']:9s} {status:10s} epoch={epoch!s:>3s} best={best!s:>3s} loss={loss}")
    completion_path = args.grid / "completion.json"
    if completion_path.is_file():
        exits = json.loads(completion_path.read_text())["exit_codes"]
        print("  exit codes:", exits)

    records = sorted(args.references.glob("references/*/record.json"))
    counts = {"accepted": 0, "hard_limit_unsettled": 0, "other": 0}
    runtimes = []
    for path in records:
        record = json.loads(path.read_text())
        status = record.get("status")
        counts[status if status in counts else "other"] += 1
        runtimes.append(record.get("wall_seconds", 0.0))
    planned = args.scene_count * 2
    remaining = max(planned - len(records), 0)
    mean = sum(runtimes) / len(runtimes) if runtimes else 0.0
    print("Sparse 192/256 references")
    print(
        f"  complete={len(records)}/{planned} accepted={counts['accepted']} "
        f"unsettled={counts['hard_limit_unsettled']} other={counts['other']} "
        f"mean_solver_wall={mean:.1f}s remaining={remaining}"
    )


if __name__ == "__main__":
    main()
