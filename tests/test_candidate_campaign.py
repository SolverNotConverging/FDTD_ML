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
