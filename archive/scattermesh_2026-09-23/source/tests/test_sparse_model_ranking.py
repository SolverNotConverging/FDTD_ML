import importlib.util
import json
from pathlib import Path

import pytest


def _ranker():
    path = Path(__file__).resolve().parents[1] / "scripts/rank_sparse_model_grid.py"
    spec = importlib.util.spec_from_file_location("sparse_model_ranker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    examples = [
        {"sample_id": "pair_val", "family": "sparse_pair", "split": "validation", "cells_x": 32},
        {"sample_id": "circle_val", "family": "sparse_cluster", "split": "validation", "cells_x": 48},
        {"sample_id": "pair_test", "family": "sparse_pair", "split": "test", "cells_x": 32},
        {"sample_id": "circle_test", "family": "sparse_cluster", "split": "test", "cells_x": 48},
    ]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"dataset_id": "test-dataset", "examples": examples}))
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({
        "models": {
            f"m{index}": {"parameter_count": 100 + index, "checkpoint_sha256": str(index)}
            for index in range(9)
        }
    }))
    reports = tmp_path / "reports"
    for index in range(9):
        cases = []
        for example in examples:
            circle = example["family"] == "sparse_cluster"
            cases.append({
                "sample_id": example["sample_id"],
                "family": example["family"],
                "scene_family_id": "three_mixed_circle_pec" if circle else "dd_circles",
                "pec_circle_count": int(circle),
                "split": example["split"],
                "cells": example["cells_x"],
                "status": "accepted",
                "improvement_over_uniform": 2.0 if index == 0 else 1.1,
            })
        output = reports / f"m{index}"
        output.mkdir(parents=True)
        (output / "report.json").write_text(json.dumps({"cases": cases}))
    return comparison, reports, dataset


def test_ranking_uses_validation_and_keeps_circular_pec_family(tmp_path):
    comparison, reports, dataset = _fixture(tmp_path)
    ranker = _ranker()
    before = ranker.rank(comparison, reports, dataset)
    assert before["selected_model"] == "m0"
    assert before["models"]["m0"]["validation"]["by_pec_circle_count"]["1"]["case_count"] == 1

    path = reports / "m8" / "report.json"
    report = json.loads(path.read_text())
    for row in report["cases"]:
        if row["split"] == "test":
            row["improvement_over_uniform"] = 1e9
    path.write_text(json.dumps(report))
    after = ranker.rank(comparison, reports, dataset)
    assert after["ranked_models"] == before["ranked_models"]
    assert after["selected_model"] == before["selected_model"]


def test_ranking_rejects_missing_cluster_validation_case(tmp_path):
    comparison, reports, dataset = _fixture(tmp_path)
    path = reports / "m0" / "report.json"
    report = json.loads(path.read_text())
    report["cases"] = [row for row in report["cases"] if row["sample_id"] != "circle_val"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="coverage differs from dataset"):
        _ranker().rank(comparison, reports, dataset)


def test_comparison_waits_for_all_nine_reports(tmp_path):
    comparison, reports, _ = _fixture(tmp_path)
    ranker = _ranker()
    assert ranker.comparison_ready(comparison, reports)[0] is True
    (reports / "m8" / "report.json").unlink()
    assert ranker.comparison_ready(comparison, reports)[0] is False
    comparison.unlink()
    assert ranker.comparison_ready(comparison, reports)[0] is False
