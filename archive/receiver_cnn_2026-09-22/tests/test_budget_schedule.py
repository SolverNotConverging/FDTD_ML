import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_training import teacher_fixture

from fdtdmesh.budget_schedule import (
    FIXED_BUDGETS,
    HELDOUT_BUDGETS,
    mixed_budget_plan,
    normalize_budgets,
)
from fdtdmesh.ml import ResUNet
from fdtdmesh.teacher_campaign import prepare_targets
from fdtdmesh.training import (
    TeacherDataset,
    TrainingConfig,
    evaluate_imitation,
    load_teacher_targets,
)


def test_plan_is_order_independent_rectangular_and_globally_holds_out_budgets():
    scenes = [
        SimpleNamespace(
            scene_id=f"scene-{i}",
            seed=i + 20,
            split="train" if i < 300 else "validation" if i < 400 else "test_iid",
        )
        for i in range(450)
    ]
    plan = mixed_budget_plan(scenes)
    assert plan == mixed_budget_plan(list(reversed(scenes)))
    assert mixed_budget_plan(scenes[:1])["assignments"]["scene-0"] == plan["assignments"]["scene-0"]
    assert len(plan["assignments"]) == 400
    rectangular = set()
    for scene in scenes[:400]:
        entries = plan["assignments"][scene.scene_id]
        pairs = [tuple(e["budget"]) for e in entries]
        assert len(pairs) == len(set(pairs)) == (8 if scene.split == "train" else 12)
        assert all(48 <= n <= 128 for pair in pairs for n in pair)
        assert set(FIXED_BUDGETS) <= set(pairs)
        rect = [tuple(e["budget"]) for e in entries if e["group"] == "rectangular"]
        assert rect[0] == rect[1][::-1] and rect[0][0] != rect[0][1]
        rectangular.update(rect)
        if scene.split == "train":
            assert not set(pairs) & set(HELDOUT_BUDGETS)
        else:
            assert set(HELDOUT_BUDGETS) <= set(pairs)
    assert len(rectangular) > 200


def test_budget_pair_normalization():
    assert normalize_budgets([48, [64, 96], [96, 64]]) == ((48, 48), (64, 96), (96, 64))
    for values in (
        [],
        [48, [48, 48]],
        [[48]],
        [[48, 64, 96]],
        [[48, 0]],
        [[True, 64]],
        [[48.5, 64]],
    ):
        with pytest.raises(ValueError):
            normalize_budgets(values)


def test_rectangular_cache_and_plan_resume_binding(tmp_path):
    manifest, _ = teacher_fixture(tmp_path)
    records = json.loads(manifest.read_text())["scenes"]
    plan = dict(
        assignments={
            s["scene_id"]: [
                dict(budget=[32, 40], group="rectangular"),
                dict(
                    budget=[32, 48],
                    group="heldout_budget" if s["split"] == "validation" else "rectangular",
                ),
                dict(budget=[40, 32], group="rectangular"),
            ]
            for s in records
        }
    )
    output = tmp_path / "rectangular"
    targets = prepare_targets(manifest, output, workers=2, budget_plan=plan)
    _, _, metadata, x, y = load_teacher_targets(manifest, targets)
    assert len(list((output / "target_records").glob("*.json"))) == 6
    assert len(metadata["samples"]) + len(metadata["failures"]) == 6
    assert {tuple(s["budget"]) for s in metadata["samples"]} == {(32, 40), (32, 48), (40, 32)}
    np.testing.assert_allclose(x.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(y.sum(1), 1, atol=1e-6)
    timestamp = targets.stat().st_mtime_ns
    prepare_targets(manifest, output, workers=2, budget_plan=plan)
    assert targets.stat().st_mtime_ns == timestamp
    dataset = TeacherDataset(manifest, targets, "validation")
    stats = evaluate_imitation(ResUNet(2), dataset, TrainingConfig(width=2), "cpu", project=False)
    assert stats["by_budget_group"]["heldout_budget"]["samples"] == 1
    assert stats["by_budget_group"]["rectangular"]["samples"] == 2
    plan["assignments"][records[0]["scene_id"]][0]["budget"] = [32, 41]
    with pytest.raises(ValueError, match="identity changed"):
        prepare_targets(manifest, output, workers=2, budget_plan=plan)
