import importlib.util
import json
from collections import Counter
from pathlib import Path

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
