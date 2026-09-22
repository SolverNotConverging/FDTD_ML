import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scattermesh.generalization import (
    GENERALIZATION_ANGLES,
    GENERALIZATION_BUDGETS,
    GENERALIZATION_CENTERS,
    GENERALIZATION_MATERIALS,
    GENERALIZATION_RADII,
    PML_THICKNESS,
    circle_generalization_plan,
)


def test_circle_generalization_plan_is_factorial_frozen_and_outside_pml():
    plan = circle_generalization_plan()
    examples = plan["examples"]
    expected = (
        len(GENERALIZATION_RADII)
        * len(GENERALIZATION_CENTERS)
        * len(GENERALIZATION_MATERIALS)
        * len(GENERALIZATION_ANGLES)
        * len(GENERALIZATION_BUDGETS)
    )
    assert len(examples) == expected == 120
    assert len({row["sample_id"] for row in examples}) == expected
    assert Counter(row["size_regime"] for row in examples) == {
        name: 40 for name in GENERALIZATION_RADII
    }
    assert Counter(row["position_id"] for row in examples) == {
        name: 24 for name in GENERALIZATION_CENTERS
    }
    assert {row["cells_x"] for row in examples} == set(GENERALIZATION_BUDGETS)
    assert {row["incidence_angle_rad"] for row in examples} == set(GENERALIZATION_ANGLES)
    coarse_cell = 1.2 / min(GENERALIZATION_BUDGETS)
    for row in examples:
        x, y = row["center_m"]
        radius = row["radius_m"]
        clearances = (
            x - radius - PML_THICKNESS,
            y - radius - PML_THICKNESS,
            1.2 - PML_THICKNESS - x - radius,
            1.2 - PML_THICKNESS - y - radius,
        )
        assert min(clearances) >= 2.5 * coarse_cell
        assert row["cells_x"] == row["cells_y"]
    assert plan["gate"] == {
        "minimum_meaningful_win_fraction": 0.70,
        "minimum_each_size_regime_win_fraction": 0.50,
        "minimum_each_position_median_improvement": 1.0,
        "minimum_overall_median_improvement": 1.2,
        "minimum_worst_case_improvement": 0.5,
        "require_all_cases_settled": True,
    }


def test_checked_in_circle_generalization_plan_matches_generator():
    path = Path(__file__).resolve().parents[1] / "configs/circle_position_scale_generalization.json"
    generated = json.loads(json.dumps(circle_generalization_plan()))
    assert json.loads(path.read_text()) == generated


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_circle_generalization_evaluation.py"
    spec = importlib.util.spec_from_file_location("circle_generalization_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_continuation():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/continue_circle_generalization_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("circle_generalization_continuation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generalization_runner_projects_exact_paired_meshes_and_summarizes(tmp_path):
    torch = pytest.importorskip("torch")
    from scattermesh.model import AxisDensityUNet

    runner = load_runner()
    model = AxisDensityUNet(base_channels=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_kwargs": {"base_channels": 8},
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    plan = circle_generalization_plan()
    plan["examples"] = plan["examples"][:2]
    cases = runner.mesh_cases(plan, checkpoint)
    assert len(cases) == 4
    assert {case["mesh_kind"] for case in cases} == {"uniform", "cnn"}
    assert all(len(case["x"]) == case["example"]["cells_x"] + 1 for case in cases)
    assert all(len(case["y"]) == case["example"]["cells_y"] + 1 for case in cases)
    assert all(np.isfinite(case["x"]).all() and np.isfinite(case["y"]).all() for case in cases)

    output = tmp_path / "output"
    for case in cases:
        learned = case["mesh_kind"] == "cnn"
        record = {
            "accepted": True,
            "joint_scattering_loss": 0.1 if learned else 0.2,
            "complex_mse_loss": 0.08 if learned else 0.16,
            "rcs_log_loss": 0.08 if learned else 0.16,
            "Nt": 100,
            "config": {
                "x_uniform_repair_fraction": case["x_uniform_repair_fraction"],
                "y_uniform_repair_fraction": case["y_uniform_repair_fraction"],
            },
        }
        path = output / "cases" / case["case_id"] / "record.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))

    report = runner.summarize(plan, cases, output)
    assert report["decision"] == "passes_circle_position_scale_generalization"
    assert all(report["checks"].values())
    assert report["overall"]["median_improvement_over_uniform"] == pytest.approx(2.0)
    assert (output / "evaluation_summary.png").is_file()


def test_generalization_continuation_waits_for_passing_main_gate(tmp_path):
    continuation = load_continuation()
    plan = tmp_path / "plan.json"
    checkpoint = tmp_path / "checkpoint.pt"
    physics = tmp_path / "physics"
    plan.write_text(json.dumps({"examples": [{"sample_id": "a"}]}))
    checkpoint.write_bytes(b"checkpoint")

    waiting = continuation.prerequisite_snapshot(plan, checkpoint, physics)
    assert waiting["ready"] is False
    physics.mkdir()
    (physics / "report.json").write_text(
        json.dumps({"decision": "fails_frozen_physics_evaluation"})
    )
    assert continuation.prerequisite_snapshot(plan, checkpoint, physics)["ready"] is False
    (physics / "report.json").write_text(
        json.dumps({"decision": "passes_frozen_physics_evaluation"})
    )
    assert continuation.prerequisite_snapshot(plan, checkpoint, physics)["ready"] is True
    assert continuation.expected_case_ids(plan) == ["a_uniform", "a_cnn"]
