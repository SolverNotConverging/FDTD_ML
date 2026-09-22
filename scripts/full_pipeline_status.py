#!/usr/bin/env python3
"""Report the status of the full label, training, and frozen-physics pipeline."""

import argparse
import importlib.util
import json
from pathlib import Path

STATUS_SCRIPT = Path(__file__).with_name("candidate_campaign_status.py")
STATUS_SPEC = importlib.util.spec_from_file_location("candidate_campaign_status", STATUS_SCRIPT)
STATUS_MODULE = importlib.util.module_from_spec(STATUS_SPEC)
STATUS_SPEC.loader.exec_module(STATUS_MODULE)
campaign_snapshot = STATUS_MODULE.snapshot


def _json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _training(path):
    workflow = _json(Path(path) / "workflow.json") or {}
    summary = _json(Path(path) / "summary.json")
    progress = _json(Path(path) / "progress.json")
    state = summary or progress or {}
    result = {"workflow": workflow.get("stage", "not_started")}
    for key in (
        "status",
        "epochs_completed",
        "planned_epochs",
        "best_epoch",
        "best_validation_loss",
        "wall_seconds",
    ):
        if key in state:
            result[key] = state[key]
    if progress is not None and "epoch" in progress:
        result["current_epoch"] = progress["epoch"]
        latest = progress.get("latest", {})
        for key in ("train_loss", "validation_loss"):
            if key in latest:
                result[key] = latest[key]
    return result


def _physics(path):
    workflow = _json(Path(path) / "workflow.json") or {}
    report = _json(Path(path) / "report.json") or {}
    result = {"workflow": workflow.get("stage", "not_started")}
    if report:
        result["report_stage"] = report.get("decision", "present")
    statuses = report.get("status_counts", {})
    result["planned"] = workflow.get("planned", report.get("case_count"))
    result["completed"] = workflow.get("completed", report.get("case_count"))
    result["accepted"] = workflow.get("accepted", statuses.get("accepted"))
    result["unsettled"] = workflow.get("unsettled", statuses.get("unsettled"))
    result["malformed"] = workflow.get("malformed")
    result["remaining"] = workflow.get("remaining")
    result = {key: value for key, value in result.items() if value is not None}
    return result


def status(campaign, campaign_output, training_output, physics_output, window_minutes=10):
    try:
        snapshot = campaign_snapshot(campaign, campaign_output, window_minutes)
        statuses = snapshot.get("simulation_status_counts", {})
        accepted = statuses.get("accepted", 0)
        unsettled = statuses.get("unsettled", 0)
        campaign_info = {
            "planned": snapshot["planned"],
            "completed": snapshot["completed"],
            "observed_records": snapshot.get("observed_records", snapshot["completed"]),
            "accepted": accepted,
            "unsettled": unsettled,
            "retrying": sum(snapshot.get("retrying_status_counts", {}).values()),
            "other_outcomes": snapshot["completed"] - accepted - unsettled,
            "malformed": snapshot.get("malformed_expected_records", 0),
            "remaining": snapshot["remaining"],
            "rate_per_hour": snapshot.get("recent_completion_rate_per_hour"),
            "eta_seconds": snapshot.get("estimated_remaining_seconds"),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        campaign_info = {
            "planned": 0,
            "completed": 0,
            "observed_records": 0,
            "accepted": 0,
            "unsettled": 0,
            "retrying": 0,
            "malformed": 0,
            "remaining": 0,
            "rate_per_hour": None,
            "eta_seconds": None,
            "status": "unavailable",
        }
    training = _training(training_output)
    physics = _physics(physics_output)
    stage = (_json(Path(training_output) / "workflow.json") or {}).get("stage", "")
    physics_stage = (_json(Path(physics_output) / "workflow.json") or {}).get("stage", "")
    report = _json(Path(physics_output) / "report.json") or {}
    failed = (
        "failed" in stage
        or "failed" in physics_stage
        or str(report.get("decision", "")).startswith("fails_")
    )
    if failed:
        active_stage = "failed"
    elif report.get("decision") == "passes_frozen_physics_evaluation" or physics_stage == "complete":
        active_stage = "complete"
    elif campaign_info.get("remaining", 0) > 0:
        active_stage = "label_generation"
    elif physics_stage in {"physics_evaluation", "summarizing"} or report:
        active_stage = "frozen_physics"
    elif stage == "training" or training.get("status") in {"running", "complete"}:
        active_stage = "training"
    elif campaign_info.get("completed", 0) == 0:
        active_stage = "label_generation"
    else:
        active_stage = "label_gate_or_dataset"
    return {"campaign": campaign_info, "training": training, "physics": physics, "active_stage": active_stage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/simple_factorial_exact_candidate_full.json"),
    )
    parser.add_argument(
        "--campaign-output",
        type=Path,
        default=Path("runs/simple_factorial_exact_candidate_full_6916879"),
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("runs/simple_factorial_exact_training_6916879"),
    )
    parser.add_argument(
        "--physics-output",
        type=Path,
        default=Path("runs/simple_factorial_exact_physics_6916879"),
    )
    parser.add_argument("--window-minutes", type=float, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            status(
                args.campaign,
                args.campaign_output,
                args.training_output,
                args.physics_output,
                args.window_minutes,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
