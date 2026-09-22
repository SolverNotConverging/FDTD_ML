#!/usr/bin/env python3
"""Rank nine sparse CNNs on held-out validation physics without using test data."""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path


def _geometric_mean(values):
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _family(row):
    return row.get("scene_family_id") or row.get("material_topology") or row["family"]


def _summarize(rows):
    by_family = defaultdict(list)
    by_pec_circles = defaultdict(list)
    for row in rows:
        by_family[_family(row)].append(row)
        by_pec_circles[str(row.get("pec_circle_count", "unspecified"))].append(row)

    def score(group):
        accepted = [
            row["improvement_over_uniform"] for row in group
            if row["status"] == "accepted"
        ]
        return {
            "case_count": len(group),
            "accepted_count": len(accepted),
            "geometric_mean_improvement": _geometric_mean(accepted),
            "meaningful_win_fraction": sum(value >= 1.05 for value in accepted) / len(group),
        }

    family = {name: score(group) for name, group in sorted(by_family.items())}
    complete_family_scores = [
        item["geometric_mean_improvement"] for item in family.values()
        if item["accepted_count"] == item["case_count"]
    ]
    all_accepted = all(row["status"] == "accepted" for row in rows)
    return {
        **score(rows),
        "all_accepted": all_accepted,
        "family_balanced_geometric_mean_improvement": (
            _geometric_mean(complete_family_scores)
            if all_accepted else None
        ),
        "by_scene_family": family,
        "by_pec_circle_count": {
            name: score(group) for name, group in sorted(by_pec_circles.items())
        },
    }


def rank(comparison_path, report_dir, dataset_path, *, low_budgets=(32, 48)):
    comparison = json.loads(Path(comparison_path).read_text())
    dataset = json.loads(Path(dataset_path).read_text())
    models = comparison["models"]
    if len(models) != 9:
        raise ValueError("Expected exactly nine model comparisons")
    expected = {
        split: {
            row["sample_id"] for row in dataset["examples"]
            if row["family"] in {"sparse_pair", "sparse_cluster"}
            and row["split"] == split and row["cells_x"] in low_budgets
        }
        for split in ("validation", "test")
    }
    if not expected["validation"] or not expected["test"]:
        raise ValueError("Dataset lacks held-out low-budget sparse physics cases")
    results = {}
    for name in sorted(models):
        report = json.loads((Path(report_dir) / name / "report.json").read_text())
        cases = report["cases"]
        if len({row["sample_id"] for row in cases}) != len(cases):
            raise ValueError(f"Duplicate physics case in {name}")
        for split in ("validation", "test"):
            actual = {
                row["sample_id"] for row in cases
                if row["split"] == split and row["cells"] in low_budgets
            }
            if actual != expected[split]:
                raise ValueError(f"Low-budget {split} coverage differs from dataset for {name}")
        validation = [row for row in cases if row["sample_id"] in expected["validation"]]
        test = [
            row for row in cases
            if row["sample_id"] in expected["test"]
        ]
        results[name] = {
            "validation": _summarize(validation),
            "test_audit": _summarize(test),
            "parameter_count": models[name]["parameter_count"],
            "checkpoint_sha256": models[name]["checkpoint_sha256"],
        }
    ordered = sorted(
        results,
        key=lambda name: (
            results[name]["validation"]["accepted_count"],
            results[name]["validation"]["family_balanced_geometric_mean_improvement"]
            or -math.inf,
            -results[name]["parameter_count"],
        ),
        reverse=True,
    )
    winner = results[ordered[0]]["validation"]
    selected = ordered[0] if (
        winner["all_accepted"]
        and winner["family_balanced_geometric_mean_improvement"] is not None
    ) else None
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "dataset_id": dataset.get("dataset_id"),
        "low_budgets": list(low_budgets),
        "ranking_rule": (
            "accepted validation case count, then equal-family geometric mean "
            "improvement over uniform, then smaller parameter count"
        ),
        "ranked_models": ordered,
        "selected_model": selected,
        "models": results,
    }


def comparison_ready(comparison_path, report_dir):
    comparison_path, report_dir = Path(comparison_path), Path(report_dir)
    if not comparison_path.is_file():
        return False, "waiting for nine-model comparison"
    models = json.loads(comparison_path.read_text()).get("models", {})
    if len(models) < 9:
        return False, f"waiting for model reports ({len(models)}/9)"
    if len(models) > 9:
        raise ValueError("Comparison contains more than nine models")
    missing = [name for name in models if not (report_dir / name / "report.json").is_file()]
    if missing:
        return False, f"waiting for reports: {', '.join(sorted(missing))}"
    return True, "nine-model comparison complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-budgets", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.poll_seconds < 0:
        raise ValueError("Poll interval cannot be negative")
    if args.poll_seconds:
        while True:
            ready, message = comparison_ready(args.comparison, args.reports)
            print(message, flush=True)
            if ready:
                break
            time.sleep(args.poll_seconds)
    result = rank(args.comparison, args.reports, args.dataset,
                  low_budgets=tuple(args.low_budgets))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    chosen = result["selected_model"]
    if chosen is None:
        print("No model accepted every low-budget validation case; inspect ranked_models")
    else:
        score = result["models"][chosen]["validation"]
        print(f"selected={chosen} accepted={score['accepted_count']}/{score['case_count']} "
              f"family_balanced_geomean={score['family_balanced_geometric_mean_improvement']}")


if __name__ == "__main__":
    main()
