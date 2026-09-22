#!/usr/bin/env python3
"""Print compact progress for sparse references, candidates, and CNN evaluation."""

import argparse
import hashlib
import json
from pathlib import Path


def print_campaign_progress(label, config_path, output):
    config = json.loads(config_path.read_text())
    phases = (
        ("references", len(config["scenes"]) * len(config["reference_budgets"])),
        (
            "candidates",
            len(config["scenes"]) * len(config["budgets"]) * len(config["policies"]),
        ),
    )
    print(label)
    for phase, planned in phases:
        records = sorted(output.glob(f"{phase}/*/record.json"))
        counts = {"accepted": 0, "hard_limit_unsettled": 0, "other": 0}
        runtimes = []
        for path in records:
            record = json.loads(path.read_text())
            status = record.get("status")
            counts[status if status in counts else "other"] += 1
            runtimes.append(record.get("wall_seconds", 0.0))
        mean = sum(runtimes) / len(runtimes) if runtimes else 0.0
        print(
            f"  {phase}: complete={len(records)}/{planned} accepted={counts['accepted']} "
            f"unsettled={counts['hard_limit_unsettled']} other={counts['other']} "
            f"mean_solver_wall={mean:.1f}s remaining={max(planned - len(records), 0)}"
        )
    report_path = output / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        print(f"  decision={report['decision']}")


def print_cluster_preflight(config_path, output):
    print("Sparse cluster mesh preflight")
    path = output / "cutcell_preflight.json"
    if not path.is_file():
        print("  waiting for cut-cell cost preflight")
        return
    report = json.loads(path.read_text())
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if report.get("config_sha256") != config_hash:
        print("  stale: cluster configuration changed after preflight")
        return
    rows = report["cases"]
    infeasible = sum(not row["mesh_feasible"] for row in rows)
    cap = [
        sum(row["mesh_feasible"] and not row["within_step_limit"][index] for row in rows)
        for index in range(len(report["duration_schedule_s"]))
    ]
    circle_refs = [
        row for row in rows
        if row["phase"] == "references" and row["pec_circle_count"] > 0
    ]
    circle_last_capped = sum(
        row["mesh_feasible"] and not row["within_step_limit"][-1]
        for row in circle_refs
    )
    print(
        f"  meshes={len(rows)} infeasible={infeasible} "
        f"step_cap_by_retry={cap} circular_PEC_references_last_retry_capped="
        f"{circle_last_capped}/{len(circle_refs)}"
    )


def print_training_progress(label, grid):
    print(label)
    launch_path = grid / "launch.json"
    if not launch_path.is_file():
        print("  waiting for training launch")
        return []
    entries = json.loads(launch_path.read_text())["entries"]
    for entry in entries:
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
    completion_path = grid / "completion.json"
    if completion_path.is_file():
        print("  exit codes:", json.loads(completion_path.read_text())["exit_codes"])
    return entries


def print_physics_progress(label, entries, physics):
    print(label)
    if not entries:
        print("  waiting for training launch")
        return
    for entry in entries:
        case_records = list((physics / entry["name"] / "cases").glob("*/record.json"))
        report_path = physics / entry["name"] / "report.json"
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


def print_selection_progress(label, physics):
    print(label)
    selection_path = physics / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        chosen = selection["selected_model"]
        if chosen is None:
            print("  no fully accepted validation model")
        else:
            validation = selection["models"][chosen]["validation"]
            print(
                f"  selected={chosen} validation={validation['accepted_count']}/"
                f"{validation['case_count']} family_geomean="
                f"{validation['family_balanced_geometric_mean_improvement']:.3f}"
            )
    else:
        comparison_path = physics / "comparison.json"
        if comparison_path.is_file():
            count = len(json.loads(comparison_path.read_text()).get("models", {}))
            print(f"  waiting for ranking; model reports={count}/9")
        else:
            print("  waiting for physics comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("runs/sparse_nine_model_grid"))
    parser.add_argument("--references", type=Path, default=Path("runs/sparse_pair_campaign_96"))
    parser.add_argument("--campaign-config", type=Path, default=Path("configs/sparse_pair_campaign_96.json"))
    parser.add_argument("--clusters", type=Path, default=Path("runs/sparse_cluster_pilot_32"))
    parser.add_argument("--cluster-config", type=Path, default=Path("configs/sparse_cluster_pilot_32.json"))
    parser.add_argument("--pec-circles", type=Path, default=Path("runs/pec_circle_gap_pilot"))
    parser.add_argument("--pec-circle-config", type=Path, default=Path("configs/pec_circle_gap_pilot.json"))
    parser.add_argument("--physics", type=Path, default=Path("runs/sparse_nine_model_physics"))
    parser.add_argument("--finetune-grid", type=Path, default=Path("runs/sparse_nine_model_finetune_96"))
    parser.add_argument("--finetune-physics", type=Path, default=Path("runs/sparse_nine_model_finetune_physics_96"))
    parser.add_argument("--cluster-grid", type=Path, default=Path("runs/sparse_nine_model_clusters"))
    parser.add_argument("--cluster-physics", type=Path, default=Path("runs/sparse_nine_model_clusters_physics"))
    args = parser.parse_args()
    initial_entries = print_training_progress("Nine-model training", args.grid)

    print_campaign_progress("Sparse pair campaign", args.campaign_config, args.references)
    print_campaign_progress("Sparse cluster pilot", args.cluster_config, args.clusters)
    print_cluster_preflight(args.cluster_config, args.clusters)
    if args.pec_circle_config.is_file():
        print_campaign_progress("Circular PEC gap pilot", args.pec_circle_config, args.pec_circles)

    print_physics_progress("Nine-model held-out physics", initial_entries, args.physics)
    print_selection_progress("Initial model ranking", args.physics)
    pair_entries = print_training_progress("Pair-data fine-tune", args.finetune_grid)
    print_physics_progress("Pair-data held-out physics", pair_entries, args.finetune_physics)
    print_selection_progress("Pair-data model ranking", args.finetune_physics)
    cluster_entries = print_training_progress("Multi-object fine-tune", args.cluster_grid)
    print_physics_progress("Multi-object held-out physics", cluster_entries, args.cluster_physics)
    print_selection_progress("Multi-object model ranking", args.cluster_physics)


if __name__ == "__main__":
    main()
