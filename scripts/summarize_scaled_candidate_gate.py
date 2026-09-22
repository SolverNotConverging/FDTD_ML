#!/usr/bin/env python3
"""Combine training-selection and frozen held-out labels into one gate audit."""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_simple_candidate_campaign.py"
SPEC = importlib.util.spec_from_file_location("scattermesh_candidate_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-output", type=Path, default=Path("runs/simple_candidate_scaled_train")
    )
    parser.add_argument(
        "--holdout-output", type=Path, default=Path("runs/simple_candidate_scaled_holdout")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/simple_candidate_scaled_gate.json")
    )
    return parser.parse_args()


def load_campaign(output):
    report = json.loads((output / "report.json").read_text())
    labels = json.loads((output / "labels.json").read_text())
    if report["decision"] != "accepted" or report["invalid_uniform_groups"]:
        raise ValueError(f"Campaign output is incomplete: {output}")
    if labels["schema_version"] != 2:
        raise ValueError(f"Unsupported label schema in {output}")
    return report, labels["groups"]


def main():
    args = parse_args()
    training_report, training_labels = load_campaign(args.training_output)
    holdout_report, holdout_labels = load_campaign(args.holdout_output)
    if training_report["dataset_id"] != holdout_report["dataset_id"]:
        raise ValueError("Training and held-out campaigns use different datasets")
    overlap = training_labels.keys() & holdout_labels.keys()
    if overlap:
        raise ValueError(f"Training and held-out labels overlap: {sorted(overlap)[:3]}")
    labels = {**training_labels, **holdout_labels}
    diversity = RUNNER.label_diversity(labels)
    readiness = RUNNER.training_readiness(diversity, True)
    statuses = Counter(training_report["status_counts"])
    statuses.update(holdout_report["status_counts"])
    result = {
        "schema_version": 1,
        "dataset_id": training_report["dataset_id"],
        "training_campaign_id": training_report["campaign_id"],
        "holdout_campaign_id": holdout_report["campaign_id"],
        "case_count": training_report["case_count"] + holdout_report["case_count"],
        "status_counts": dict(sorted(statuses.items())),
        "illumination_group_count": len(labels),
        "label_diversity": diversity,
        "training_readiness": readiness,
        "decision": (
            "ready_for_factorial_candidate_pilot"
            if readiness["decision"] == "ready_for_m5_pilot"
            else "repeat_candidate_search"
        ),
    }
    RUNNER.PILOT.atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
