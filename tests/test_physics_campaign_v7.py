import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.generate import GenerationConfig, make_scene
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.physics import _search_budget_entries, search_physics_targets


def campaign_module():
    path = Path(__file__).resolve().parents[1] / "scripts/start_physics_campaign_v7.py"
    spec = importlib.util.spec_from_file_location("physics_campaign_v7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign_scene(family, split, index):
    return SimpleNamespace(
        family=family,
        split=split,
        scene_id=f"{split}-{family}-{index:02d}",
    )


def test_search_budget_entries_preserve_square_and_rectangular_assignments():
    scenes = [SimpleNamespace(scene_id="scene-a"), SimpleNamespace(scene_id="scene-b")]
    plan = {
        "assignments": {
            "scene-a": [
                {"budget": 64, "group": "fixed_square"},
                {"budget": [72, 104], "group": "rectangular"},
            ],
            "scene-b": [{"budget": [104, 72], "group": "rectangular"}],
        }
    }

    entries = _search_budget_entries(scenes, budget_plan=plan)

    assert entries == {
        "scene-a": [
            {"budget": [64, 64], "group": "fixed_square"},
            {"budget": [72, 104], "group": "rectangular"},
        ],
        "scene-b": [{"budget": [104, 72], "group": "rectangular"}],
    }


@pytest.mark.parametrize(
    "scenes, kwargs, message",
    [
        (
            [SimpleNamespace(scene_id="scene-a")],
            {"budget_plan": {"assignments": {}}},
            "Missing budget assignments",
        ),
        (
            [SimpleNamespace(scene_id="scene-a")],
            {
                "budget_plan": {
                    "assignments": {
                        "scene-a": [
                            {"budget": [64, 64]},
                            {"budget": [64, 64]},
                        ]
                    }
                }
            },
            "distinct",
        ),
        (
            [SimpleNamespace(scene_id="scene-a")],
            {
                "budgets": [64],
                "budget_plan": {
                    "assignments": {"scene-a": [{"budget": [64, 64]}]}
                },
            },
            "Choose budgets or",
        ),
    ],
)
def test_search_budget_entries_reject_invalid_budget_plans(scenes, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _search_budget_entries(scenes, **kwargs)


def test_search_missing_teacher_writes_durable_failure_for_each_pair_and_resumes(
    tmp_path, monkeypatch
):
    config = GenerationConfig(
        min_objects=1, max_objects=1, raster_size=32, size_min=0.125, size_max=0.3
    )
    scene = make_scene(13, "train", config=config)
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, [scene], generation={"test": True})
    teacher_path = tmp_path / "teacher.npz"
    checkpoint_path = tmp_path / "checkpoint.pt"
    teacher_path.write_bytes(b"teacher")
    checkpoint_path.write_bytes(b"checkpoint")
    plan = {
        "assignments": {
            scene.scene_id: [
                {"budget": [64, 64], "group": "square"},
                {"budget": [72, 104], "group": "rectangular"},
            ]
        }
    }
    monkeypatch.setattr("fdtdmesh.physics._teacher_map", lambda *args: {})
    monkeypatch.setattr("fdtdmesh.physics.load_model", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(
        "fdtdmesh.physics._reference",
        lambda *args, **kwargs: (
            {"status": "converged", "accepted_budget": [128, 128]},
            scene,
            np.linspace(0, 1, 3),
            np.array([1.0]),
            np.zeros((3, 1)),
        ),
    )
    output = tmp_path / "search"

    first = search_physics_targets(
        manifest_path,
        teacher_path,
        checkpoint_path,
        output,
        scene_ids=[scene.scene_id],
        budget_plan=plan,
        device="cpu",
    )

    assert first["targets"] == 0
    result_paths = [
        output / scene.scene_id / name / "result.json" for name in ("64_64", "72_104")
    ]
    for result_path in result_paths:
        row = json.loads(result_path.read_text())
        assert row["status"] == "failed"
        assert row["reason"] == "missing_teacher"

    def retry_forbidden(*args, **kwargs):
        pytest.fail("A durable failed pair must not retry teacher lookup")

    monkeypatch.setattr("fdtdmesh.physics._teacher_prior", retry_forbidden)
    second = search_physics_targets(
        manifest_path,
        teacher_path,
        checkpoint_path,
        output,
        scene_ids=[scene.scene_id],
        budget_plan=plan,
        device="cpu",
    )
    assert second["targets"] == 0

    changed_plan = {
        "assignments": {
            scene.scene_id: [{"budget": [96, 96], "group": "changed"}]
        }
    }
    with pytest.raises(ValueError, match="different run"):
        search_physics_targets(
            manifest_path,
            teacher_path,
            checkpoint_path,
            output,
            scene_ids=[scene.scene_id],
            budget_plan=changed_plan,
            device="cpu",
        )


def test_campaign_selection_excludes_pilot_and_test_scenes():
    campaign = campaign_module()
    scenes = [
        campaign_scene(family, split, index)
        for family in campaign.FAMILIES
        for split, count in (("train", 18), ("validation", 6), ("test_iid", 2))
        for index in range(count)
    ]
    excluded = {
        f"train-{family}-00" for family in campaign.FAMILIES
    } | {f"validation-{family}-00" for family in campaign.FAMILIES}

    selected, lanes = campaign.select(scenes, excluded)

    assert len(selected) == 20 * len(campaign.FAMILIES)
    assert len(lanes) == 4
    for family in campaign.FAMILIES:
        train = [sid for sid, row in selected.items() if row == {"split": "train", "family": family}]
        validation = [
            sid
            for sid, row in selected.items()
            if row == {"split": "validation", "family": family}
        ]
        assert len(train) == 16
        assert len(validation) == 4
        assert not set(train + validation) & excluded
        assert all("test_iid" not in sid for sid in train + validation)


def test_campaign_progress_counts_failed_pairs_and_candidate_failures(tmp_path):
    campaign = campaign_module()
    workflow = {
        "requested_pairs": 2,
        "lanes": [["scene-a"], ["scene-b"], [], []],
        "budget_plan": {
            "assignments": {
                "scene-a": [{"budget": [64, 64]}],
                "scene-b": [{"budget": [72, 104]}],
            }
        },
    }
    (tmp_path / "workflow.json").write_text(json.dumps(workflow))
    failed_path = tmp_path / "lane0/search/scene-a/64_64/result.json"
    failed_path.parent.mkdir(parents=True)
    failed_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "candidates": [
                    {"status": "failed"},
                    {"status": "failed"},
                ],
            }
        )
    )
    success_path = tmp_path / "lane1/search/scene-b/72_104/result.json"
    success_path.parent.mkdir(parents=True)
    success_path.write_text(
        json.dumps(
            {
                "sample": {"scene_id": "scene-b"},
                "candidates": [{"status": "ok"}, {"status": "failed"}],
            }
        )
    )

    report = campaign.progress(tmp_path)

    assert report["requested_pairs"] == 2
    assert report["completed_pairs"] == 2
    assert report["targets"] == 1
    assert report["failed_pairs"] == 1
    assert report["recorded_candidates"] == 4
    assert report["candidate_failures"] == 3


def test_pml_migration_reuses_only_identical_successful_pairs(tmp_path):
    from test_uniform_baseline import example
    from fdtdmesh.uniform import uniform_scene

    campaign = campaign_module()
    source, output, references = (tmp_path / name for name in ("old", "new", "refs"))
    scene = example()
    campaign.write(references / scene.scene_id / "evaluated_scene.json", scene.to_dict())
    workflow = {key: {} for key in (
        "dataset_id", "checkpoint_sha256", "teacher_targets_sha256", "selection",
        "lanes", "budget_plan", "reference_bindings", "evaluation", "search", "distillation",
    )}
    workflow.update(references=str(references), source_sha256="original-source")
    campaign.write(source / "workflow.json", workflow)
    for lane in range(4):
        campaign.write(source / f"lane{lane}/search/run.json",
                       dict(baseline_policy="uniform_snapped_pec_and_quasi_uniform_v1"))
    for n in (64, 76, 96):
        directory = source / f"lane0/search/{scene.scene_id}/{n}_{n}"
        campaign.write(directory / "result.json", dict(
            sample=dict(scene_id=scene.scene_id, budget=[n, n]), candidates=[]))
        mesh = uniform_scene(scene, [n, n]).mesh
        x = mesh.x.copy()
        if n == 96:
            x[20] += (x[21] - x[20]) * 0.1
        np.savez(directory / "uniform.npz", x=x, y=mesh.y)
        np.savez(directory / "target.npz", target_x=np.ones(2), target_y=np.ones(2))
    campaign.write(source / f"lane0/search/{scene.scene_id}/51_51/result.json",
                   dict(status="failed", error="old PML mismatch", candidates=[]))
    campaign.reuse_unchanged_pairs(output, workflow, source)
    report = campaign.read(output / "reuse.json")
    assert report["count"] == 1 and report["pairs"][0]["budget"] == [64, 64]
    imported = campaign.read(output / f"lane0/search/{scene.scene_id}/64_64/result.json")
    assert imported["cache_origin"]["source_sha256"] == "original-source"
    assert len(list(output.glob("lane*/search/*/*/result.json"))) == 1
    altered = dict(workflow, checkpoint_sha256="different-model")
    with pytest.raises(ValueError, match="changed checkpoint_sha256"):
        campaign.reuse_unchanged_pairs(output, altered, source)
