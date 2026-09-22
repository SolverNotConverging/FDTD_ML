#!/usr/bin/env python3
"""Print compact progress for sparse references, candidates, and CNN evaluation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("runs/sparse_nine_model_grid"))
    parser.add_argument("--references", type=Path, default=Path("runs/sparse_pair_campaign_96"))
    parser.add_argument("--campaign-config", type=Path, default=Path("configs/sparse_pair_campaign_96.json"))
    parser.add_argument("--physics", type=Path, default=Path("runs/sparse_nine_model_physics"))
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

    config = json.loads(args.campaign_config.read_text())
    phases = (
        ("references", len(config["scenes"]) * len(config["reference_budgets"])),
        (
            "candidates",
            len(config["scenes"]) * len(config["budgets"]) * len(config["policies"]),
        ),
    )
    for phase, planned in phases:
        records = sorted(args.references.glob(f"{phase}/*/record.json"))
        counts = {"accepted": 0, "hard_limit_unsettled": 0, "other": 0}
        runtimes = []
        for path in records:
            record = json.loads(path.read_text())
            status = record.get("status")
            counts[status if status in counts else "other"] += 1
            runtimes.append(record.get("wall_seconds", 0.0))
        mean = sum(runtimes) / len(runtimes) if runtimes else 0.0
        print(f"Sparse {phase}")
        print(
            f"  complete={len(records)}/{planned} accepted={counts['accepted']} "
            f"unsettled={counts['hard_limit_unsettled']} other={counts['other']} "
            f"mean_solver_wall={mean:.1f}s remaining={max(planned - len(records), 0)}"
        )
    campaign_report = args.references / "report.json"
    if campaign_report.is_file():
        report = json.loads(campaign_report.read_text())
        print(f"  campaign decision={report['decision']}")

    print("Nine-model held-out physics")
    for entry in launch["entries"]:
        case_records = list((args.physics / entry["name"] / "cases").glob("*/record.json"))
        report_path = args.physics / entry["name"] / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            validation = report["split_reports"]["validation"]
            print(
                f"  {entry['name']:9s} complete {len(case_records)} cases "
                f"validation={validation['accepted_count']}/{validation['case_count']} "
                f"median={validation['median_improvement']}"
            )
        elif case_records:
            print(f"  {entry['name']:9s} running {len(case_records)} cases")
        else:
            print(f"  {entry['name']:9s} waiting")


if __name__ == "__main__":
    main()
