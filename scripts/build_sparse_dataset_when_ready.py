#!/usr/bin/env python3
"""Build the expanded sparse training dataset after the full physics gate passes."""

import argparse
import json
import time
from pathlib import Path

import build_sparse_joint_dataset as dataset_builder
import run_sparse_pair_headroom_pilot as campaign

ROOT = Path(__file__).resolve().parents[1]


def campaign_ready(config_path, campaign_output):
    report_path = campaign_output / "report.json"
    if not report_path.is_file():
        return False, "waiting for candidate campaign report"
    report = json.loads(report_path.read_text())
    config = campaign.load_config(config_path)
    expected_references = len(config["scenes"]) * len(config["reference_budgets"])
    expected_candidates = len(config["scenes"]) * len(config["budgets"]) * len(config["policies"])
    if (report.get("reference_case_count") != expected_references
            or report.get("candidate_case_count") != expected_candidates):
        raise ValueError("Campaign report does not cover every declared case")
    if report["decision"] != "passes_sparse_pair_headroom" or not all(
        report["checks"].values()
    ):
        raise ValueError(f"Sparse campaign did not pass its frozen gate: {report['checks']}")
    sources = campaign.source_hashes(config_path)
    if report["source_hashes"] != {"reference_phase": sources, "candidate_phase": sources}:
        raise ValueError("Campaign report has stale numerical source hashes")
    campaign._load_records(config, campaign_output, "references", sources)
    campaign._load_records(config, campaign_output, "candidates", sources)
    return True, f"{expected_references} references and {expected_candidates} candidates verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, default=ROOT / "runs/sparse_joint_dataset_pairs/dataset.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_pair_campaign_96.json")
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/sparse_pair_campaign_96")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sparse_joint_dataset_96")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("Poll interval must be positive")
    config_path, campaign_output = args.config.resolve(), args.campaign.resolve()
    while True:
        ready, message = campaign_ready(config_path, campaign_output)
        print(message, flush=True)
        if ready or args.check_only:
            break
        time.sleep(args.poll_seconds)
    if args.check_only:
        return
    payload = dataset_builder.build(
        args.base_dataset.resolve(), [(config_path, campaign_output)], args.output.resolve()
    )
    print(json.dumps({
        "dataset_id": payload["dataset_id"],
        "example_count": payload["example_count"],
        "family_split_counts": payload["family_split_counts"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
