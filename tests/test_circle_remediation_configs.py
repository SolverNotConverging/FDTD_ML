import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scattermesh import circle_remediation_pool, simple_candidate_tasks
from scattermesh.curriculum import REMEDIATION_BUDGETS
from scattermesh.generalization import widest_non_pml_monitor_bounds

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_simple_candidate_campaign.py"
    spec = importlib.util.spec_from_file_location("remediation_campaign_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_configs():
    manifest = json.loads((ROOT / "configs/circle_remediation_pool.json").read_text())
    campaign = json.loads((ROOT / "configs/circle_remediation_candidate.json").read_text())
    return manifest, campaign


def json_shape(value):
    if isinstance(value, tuple):
        return [json_shape(item) for item in value]
    if isinstance(value, list):
        return [json_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: json_shape(item) for key, item in value.items()}
    return value


def test_manifest_matches_deterministic_generator_and_counts():
    manifest, _ = read_configs()
    geometries = circle_remediation_pool()
    conditions = simple_candidate_tasks(geometries, budgets=REMEDIATION_BUDGETS)
    assert manifest["geometries"] == json_shape(geometries)
    assert manifest["conditions"] == json_shape(conditions)
    assert manifest["geometry_count"] == 40
    assert manifest["condition_count"] == 256
    assert manifest["dataset_id"].startswith("circle_remediation_")
    assert len({row["geometry_id"] for row in geometries}) == 40
    assert len({row["task_id"] for row in conditions}) == 256
    assert Counter(row["split"] for row in geometries) == {
        "train": 24,
        "validation": 8,
        "test": 8,
    }
    assert Counter(row["split"] for row in conditions) == {
        "train": 192,
        "validation": 32,
        "test": 32,
    }


def test_campaign_matches_manifest_and_has_exact_case_policy():
    manifest, campaign = read_configs()
    runner = load_runner()
    inputs = runner.load_inputs(
        ROOT / "configs/circle_remediation_pool.json",
        ROOT / "configs/circle_remediation_candidate.json",
    )
    cases = runner.case_definitions(*inputs)
    assert campaign["geometry_ids"] == [row["geometry_id"] for row in manifest["geometries"]]
    assert campaign["condition_ids"] == [row["task_id"] for row in manifest["conditions"]]
    assert campaign["candidate_names"] == [
        "uniform",
        "interface_wide",
        "region_wide",
        "region_medium",
        "region_strong",
        "hybrid_wide",
    ]
    assert campaign["duration_schedule_s"] == [70e-9, 140e-9, 560e-9, 1.12e-6, 2.24e-6]
    assert campaign["pml_thickness_m"] == 0.12
    assert campaign["monitor_policy"] == "widest_non_pml_enclosing"
    assert campaign["ranking"] == {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1}
    assert len(cases) == 1536
    assert len({case["case_id"] for case in cases}) == 1536
    assert Counter(case["split"] for case in cases) == {
        "train": 1152,
        "validation": 192,
        "test": 192,
    }
    assert Counter(case["candidate"] for case in cases) == {
        name: 256 for name in campaign["candidate_names"]
    }


def test_every_campaign_grid_has_requested_budget_and_legal_monitor():
    manifest, campaign = read_configs()
    runner = load_runner()
    inputs = runner.load_inputs(
        ROOT / "configs/circle_remediation_pool.json",
        ROOT / "configs/circle_remediation_candidate.json",
    )
    cases = runner.case_definitions(*inputs)
    geometry_by_id = {row["geometry_id"]: row for row in manifest["geometries"]}
    for case in cases:
        grid = runner.PILOT.make_grid(case)
        assert grid.shape == (case["cells"] + 1, case["cells"] + 1)
        geometry = geometry_by_id[case["geometry_id"]]
        radius = geometry["radius_m"]
        cx, cy = geometry["center_m"]
        bounds = (cx - radius, cx + radius, cy - radius, cy + radius)
        monitor = widest_non_pml_monitor_bounds(
            grid, [bounds], campaign["pml_thickness_m"]
        )
        assert np.all(np.isfinite(monitor))
        assert monitor[0] < monitor[1] and monitor[2] < monitor[3]
