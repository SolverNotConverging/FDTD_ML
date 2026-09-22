import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_simple_candidate_campaign.py"
    spec = importlib.util.spec_from_file_location("candidate_campaign_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_campaign_expands_unique_reusable_cases():
    runner = load_runner()
    inputs = runner.load_inputs(
        ROOT / "configs/simple_dielectric_pool.json",
        ROOT / "configs/simple_candidate_pilot.json",
    )
    cases = runner.case_definitions(*inputs)
    assert len(cases) == 270
    assert len({case["case_id"] for case in cases}) == 270
    assert len({case["condition_id"] for case in cases}) == 30
    assert Counter(case["split"] for case in cases) == {
        "train": 162,
        "validation": 54,
        "test": 54,
    }
    assert all("campaign_id" not in case for case in cases)
    assert all(case["dataset_id"] == "simple_dk_3ea40e8117434e5c" for case in cases)


def test_full_campaign_covers_the_entire_manifest():
    runner = load_runner()
    manifest_path = ROOT / "configs/simple_dielectric_pool.json"
    campaign_path = ROOT / "configs/simple_candidate_full.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = runner.load_inputs(manifest_path, campaign_path)
    cases = runner.case_definitions(*inputs)
    assert len(cases) == 3168
    assert {case["condition_id"] for case in cases} == {
        condition["task_id"] for condition in manifest["conditions"]
    }


def test_candidate_variants_can_use_less_resolution_than_the_target_budget(tmp_path):
    runner = load_runner()
    manifest_path = ROOT / "configs/simple_dielectric_pool.json"
    manifest = json.loads(manifest_path.read_text())
    condition = manifest["conditions"][0]
    campaign = {
        "schema_version": 2,
        "dataset_id": manifest["dataset_id"],
        "campaign_id": "scaled_test",
        "geometry_ids": [condition["geometry_id"]],
        "condition_ids": [condition["task_id"]],
        "candidate_names": ["uniform", "region_medium_f87"],
        "candidate_variants": [
            {
                "candidate": "region_medium_f87",
                "base_candidate": "region_medium",
                "cell_factor": 0.87,
            }
        ],
        "duration_s": 7e-8,
    }
    campaign_path = tmp_path / "scaled.json"
    campaign_path.write_text(json.dumps(campaign))
    cases = runner.case_definitions(*runner.load_inputs(manifest_path, campaign_path))
    by_name = {case["candidate"]: case for case in cases}
    target = condition["cells_x"]
    assert by_name["uniform"]["cells"] == target
    assert "target_cells_x" not in by_name["uniform"]
    scaled = by_name["region_medium_f87"]
    assert scaled["cells"] == int(target * 0.87)
    assert scaled["target_cells_x"] == target
    assert scaled["candidate_cell_factor"] == 0.87
    assert scaled["base_candidate"] == "region_medium"


def test_campaign_source_hashes_are_pinned_across_resumes(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner.PILOT, "source_hashes", lambda: {"solver.py": "first"})
    assert runner.campaign_source_hashes(tmp_path) == {"solver.py": "first"}
    monkeypatch.setattr(runner.PILOT, "source_hashes", lambda: {"solver.py": "changed"})
    assert runner.campaign_source_hashes(tmp_path) == {"solver.py": "first"}
    payload = json.loads((tmp_path / "source_hashes.json").read_text())
    assert payload["source_revision"] is None


def test_label_diversity_and_training_gate_require_split_safe_variation():
    runner = load_runner()
    labels = {
        "train_a": {
            "split": "train",
            "lineage_id": "lineage_a",
            "budgets": {
                "32": {
                    "uniform_case": "u1",
                    "best_case": "a1",
                    "best_candidate": "region_medium",
                    "improvement": 1.2,
                }
            },
        },
        "train_b": {
            "split": "train",
            "lineage_id": "lineage_b",
            "budgets": {
                "48": {
                    "uniform_case": "u2",
                    "best_case": "b1",
                    "best_candidate": "hybrid_wide",
                    "improvement": 1.1,
                }
            },
        },
        "validation": {
            "split": "validation",
            "lineage_id": "lineage_v",
            "budgets": {
                "32": {
                    "uniform_case": "u3",
                    "best_case": "v1",
                    "best_candidate": "region_medium",
                    "improvement": 1.06,
                }
            },
        },
        "test": {
            "split": "test",
            "lineage_id": "lineage_t",
            "budgets": {
                "32": {
                    "uniform_case": "u4",
                    "best_case": "t1",
                    "best_candidate": "region_medium",
                    "improvement": 1.07,
                }
            },
        },
    }
    diversity = runner.label_diversity(labels)
    assert diversity["train"]["meaningful_nonuniform_wins"] == 2
    assert diversity["train"]["winning_budgets"] == {"32": 1, "48": 1}
    assert runner.training_readiness(diversity, True)["decision"] == "ready_for_m5_pilot"

    labels["train_b"]["budgets"]["48"]["improvement"] = 1.01
    diversity = runner.label_diversity(labels)
    readiness = runner.training_readiness(diversity, True)
    assert readiness["decision"] == "not_ready_for_m5"
    assert not readiness["checks"]["train_has_multiple_winning_lineages"]


def test_candidate_scorecard_uses_best_affordable_resolution():
    runner = load_runner()

    def row(candidate, cells, updates, loss, *, accepted=True):
        return {
            "accepted": accepted,
            "status": "accepted" if accepted else "unsettled",
            "cell_updates": updates,
            "joint_scattering_loss": loss,
            "config": {
                "candidate": candidate,
                "cells": cells,
                "target_cells_x": cells,
            },
        }

    groups = {
        "illumination": [
            row("uniform", 48, 100, 0.20),
            row("uniform", 64, 200, 0.10),
            row("region_f90", 48, 90, 0.15),
            row("region_f90", 64, 180, 0.08),
            row("region_f80", 48, 80, 0.25),
            row("region_f80", 64, 160, 0.09),
            row("region_f80", 32, 50, 0.30, accepted=False),
        ]
    }
    scorecard = runner.candidate_scorecard(groups)
    assert scorecard["region_f90"]["affordable_comparisons"] == 2
    assert scorecard["region_f90"]["meaningful_wins"] == 2
    assert scorecard["region_f90"]["maximum_improvement"] == pytest.approx(4 / 3)
    assert scorecard["region_f80"]["status_counts"] == {"accepted": 2, "unsettled": 1}


def test_scaled_holdout_plan_uses_only_frozen_splits_and_candidates():
    runner = load_runner()
    campaign_path = ROOT / "configs/simple_candidate_scaled_holdout.json"
    campaign = json.loads(campaign_path.read_text())
    cases = runner.case_definitions(
        *runner.load_inputs(ROOT / "configs/simple_dielectric_pool.json", campaign_path)
    )
    assert len(cases) == 320
    assert {case["split"] for case in cases} == {"validation", "test"}
    assert {case["lineage_id"] for case in cases} == {
        "simple_dk_lineage_06",
        "simple_dk_lineage_07",
    }
    assert {case["candidate"] for case in cases} == set(campaign["candidate_names"])
    assert campaign["selection_source_campaign_id"] == (
        "simple_candidate_scaled_train_b7d0de8eb710babc"
    )


def test_factorial_exact_pilot_covers_extremes_and_all_splits():
    runner = load_runner()
    manifest_path = ROOT / "configs/simple_dielectric_factorial_pool.json"
    campaign_path = ROOT / "configs/simple_factorial_exact_candidate_pilot.json"
    campaign = json.loads(campaign_path.read_text())
    cases = runner.case_definitions(*runner.load_inputs(manifest_path, campaign_path))
    assert len(cases) == 312
    assert Counter(case["split"] for case in cases) == {
        "train": 216,
        "validation": 48,
        "test": 48,
    }
    assert len({case["geometry_id"] for case in cases}) == 5
    assert {case["candidate"] for case in cases} == set(campaign["candidate_names"])
    assert all(case["cells"] in {32, 48, 64, 96} for case in cases)
    assert all("candidate_cell_factor" not in case for case in cases)
    assert all(case["duration"] == 70e-9 for case in cases)
    assert all(case["duration_schedule_s"] == [70e-9, 140e-9, 560e-9] for case in cases)
    assert campaign["ranking"] == {
        "mode": "fixed_axis_soft_nt",
        "nt_cost_exponent": 0.1,
    }


def test_factorial_exact_full_covers_every_condition_at_exact_budgets():
    runner = load_runner()
    manifest_path = ROOT / "configs/simple_dielectric_factorial_pool.json"
    campaign_path = ROOT / "configs/simple_factorial_exact_candidate_full.json"
    manifest = json.loads(manifest_path.read_text())
    campaign = json.loads(campaign_path.read_text())
    cases = runner.case_definitions(*runner.load_inputs(manifest_path, campaign_path))
    assert len(cases) == 1552 * 6
    assert {case["condition_id"] for case in cases} == {
        row["task_id"] for row in manifest["conditions"]
    }
    target_cells = {row["task_id"]: row["cells_x"] for row in manifest["conditions"]}
    assert all(case["cells"] == target_cells[case["condition_id"]] for case in cases)
    assert all("candidate_cell_factor" not in case for case in cases)
    assert all(case["duration_schedule_s"] == [70e-9, 140e-9, 560e-9] for case in cases)
    assert (
        campaign["pilot_campaign_id"]
        == "simple_factorial_exact_candidate_pilot_6fb40ec9fe0fe154"
    )


def test_fixed_axis_ranking_uses_soft_nt_penalty():
    runner = load_runner()
    baseline = {
        "accepted": True,
        "joint_scattering_loss": 0.2,
        "cell_updates": 100,
        "Nt": 100,
        "Nx": 48,
        "Ny": 48,
        "config": {"candidate": "uniform", "cells": 48},
    }
    focused = {
        "accepted": True,
        "joint_scattering_loss": 0.1,
        "cell_updates": 160,
        "Nt": 160,
        "Nx": 48,
        "Ny": 48,
        "config": {"candidate": "region_medium", "cells": 48},
    }
    wrong_budget = {
        **focused,
        "Nx": 47,
        "config": {"candidate": "region_medium", "cells": 47},
    }
    ranked = runner.ranked_candidates(
        [baseline, focused, wrong_budget],
        baseline,
        48,
        {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1},
    )
    assert [row["config"]["candidate"] for row, _ in ranked] == ["uniform", "region_medium"]
    assert ranked[1][1] == pytest.approx(0.1 * 1.6**0.1)


def test_duration_attempts_preserve_logical_case_and_increase_time():
    runner = load_runner()
    config = {
        "case_id": "logical_case",
        "duration": 70e-9,
        "duration_schedule_s": [70e-9, 140e-9, 560e-9],
    }
    attempts = runner.duration_attempt_configs(config)
    assert [row["duration"] for row in attempts] == [70e-9, 140e-9, 560e-9]
    assert [row["duration_attempt_index"] for row in attempts] == [0, 1, 2]
    assert {row["case_id"] for row in attempts} == {"logical_case"}


def test_duration_schedule_archives_unsettled_attempt_and_resumes(tmp_path, monkeypatch):
    runner = load_runner()
    config = {
        "case_id": "logical_case",
        "duration": 70e-9,
        "duration_schedule_s": [70e-9, 140e-9, 560e-9],
    }
    sources = {"solver.py": "hash"}
    calls = []

    def fake_run_case(attempt, output, device, source_hashes):
        calls.append(attempt["duration"])
        accepted = attempt["duration_attempt_index"] == 1
        record = {
            "case_id": attempt["case_id"],
            "fingerprint": runner.PILOT.sha256_json(
                {"config": attempt, "sources": source_hashes}
            ),
            "config": attempt,
            "backend": "torch_cuda",
            "accepted": accepted,
            "status": "accepted" if accepted else "unsettled",
        }
        directory = output / "cases" / attempt["case_id"]
        runner.PILOT.atomic_json(directory / "record.json", record)
        runner.PILOT.atomic_npz(
            directory / "spectra.npz",
            complex_far_field=np.ones((3, 180), dtype=np.complex128),
            analytic_complex_far_field=np.ones((3, 180), dtype=np.complex128),
        )
        return record, False

    monkeypatch.setattr(runner.PILOT, "run_case", fake_run_case)
    record, cached = runner.run_campaign_case(config, tmp_path, "cuda:0", sources)
    assert not cached
    assert record["accepted"]
    assert record["config"]["duration"] == 140e-9
    assert calls == [70e-9, 140e-9]
    archived = tmp_path / "attempts" / "logical_case" / "attempt_00"
    assert (archived / "record.json").exists()
    assert (archived / "spectra.npz").exists()

    resumed, cached = runner.run_campaign_case(config, tmp_path, "cuda:0", sources)
    assert cached
    assert resumed == record
    assert calls == [70e-9, 140e-9]
