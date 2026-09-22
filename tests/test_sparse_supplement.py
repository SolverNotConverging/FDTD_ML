import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.campaign import quotas
from fdtdmesh.data.generate_v7 import FAMILIES, SparseConfig, feature_audit, make_sparse_scene
from fdtdmesh.data.schema import near_geometry, read_manifest, validate_splits, write_manifest
from fdtdmesh.data.supplement import audit_campaign, merge_campaigns, reserve_geometry
from fdtdmesh.training import TrainingConfig, training_sample_weights


def test_sparse_scene_coverage_resolution_and_anchors():
    scenes = [make_sparse_scene(2026, "train", i) for i in range(48)]
    assert {s.family for s in scenes} == set(FAMILIES)
    assert scenes[7].content_hash == make_sparse_scene(2026, "train", 7).content_hash
    assert scenes[7].seed != make_sparse_scene(2026, "validation", 7).seed
    assert {g["kind"] for s in scenes for g in s.geometry} == {
        "rectangle",
        "circle",
        "triangle",
        "polygon",
    }
    centers = []
    for s in scenes:
        assert len(s.geometry) == (2 if s.family in ("dielectric_gap", "pec_dielectric") else 1)
        assert sum(g["material"] == "PEC" for g in s.geometry) == int("pec" in s.family)
        assert all(
            m["mu_r"] == 1 and m["sigma_h"] == 0 and 1 < m["epsilon_r"] <= 30 for m in s.materials
        )
        assert s.budgets == [[n, n] for n in (48, 64, 96, 128)]
        audit = feature_audit(s)
        assert min(min(g["cells_at_1024"]) for g in audit["objects"]) >= 81.92 - 1e-8
        if len(s.geometry) == 2:
            assert 4 - 1e-9 <= audit["gap_pixels"] <= 8 + 1e-9
        for n in (128, 256):
            sim = s.build([n, n], reference=True)
            mesh = sim.mesh_uniform()
            for axis in ("x", "y"):
                assert all(a in getattr(mesh, axis) for a in getattr(sim, axis + "_anchors"))
        sim = s.build([128, 128])
        x, y = [(np.arange(128) + 0.5) * d / 128 for d in s.domain]
        eps, _, _, pec = sim.sample(x, y)
        centers.append(np.argwhere((eps != 1) | pec).mean(0) / 128)
    assert np.min(centers) < 0.3 and np.max(centers) > 0.7


def test_sparse_background_is_not_a_duplicate_and_split_leaks_still_rejected(tmp_path):
    a = np.zeros(4096, dtype=bool)
    b = a.copy()
    a[0:4] = True
    b[100:104] = True
    assert not near_geometry(a, b)
    assert near_geometry(a, a)
    a[:200] = b[:200] = True
    b[199] = False
    assert near_geometry(a, b)
    scene = make_sparse_scene(2026, "train", 0)
    duplicate = replace(scene, split="validation", scene_id="duplicate", group_id="distinct-group")
    with pytest.raises(ValueError, match="Near-duplicate"):
        validate_splits([scene, duplicate])
    assert reserve_geometry(tmp_path, scene)
    assert reserve_geometry(tmp_path, scene)
    assert not reserve_geometry(tmp_path, duplicate)


def test_exact_sparse_family_quotas():
    for workers in (1, 3, 4):
        for target in (1, 7, 100, 800):
            totals = np.sum(
                [quotas(target, workers, lane, balanced=True) for lane in range(workers)], axis=0
            )
            assert totals.sum() == target
            assert totals.max() - totals.min() <= 1
    assert (
        list(np.sum([quotas(100, 4, lane, balanced=True) for lane in range(4)], axis=0)) == [25] * 4
    )


def campaign_fixture(root, scene):
    root.mkdir()
    manifest = write_manifest(root / "accepted_manifest.json", [scene], generation={})
    (root / "complete.json").write_text(
        json.dumps(dict(dataset_id=manifest["dataset_id"], accepted=1))
    )
    (root / "campaign.json").write_text(json.dumps(dict(targets={scene.split: 1})))
    references = root / "references"
    references.mkdir()
    (references / "reference.json").write_text(
        json.dumps(
            dict(scene_hash=scene.content_hash, status="converged", accepted_budget=[1024, 1024])
        )
    )
    np.savez(
        references / "reference_latest.npz",
        times=np.arange(4),
        waveforms=np.ones((4, 1)),
        spectra=np.ones((4, 1)),
    )
    decisions = root / "lane0" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / f"{scene.scene_id}.json").write_text(
        json.dumps(
            dict(
                scene_id=scene.scene_id,
                scene_hash=scene.content_hash,
                status="converged",
                reference="references/reference.json",
            )
        )
    )


def test_combined_corpus_keeps_scene_and_reference_identity(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    scenes = [make_sparse_scene(7, "train", 0), make_sparse_scene(7, "validation", 3)]
    campaign_fixture(a, scenes[0])
    campaign_fixture(b, scenes[1])
    output = tmp_path / "combined"
    merged = merge_campaigns([a, b], output)
    _, actual, audit = audit_campaign(output)
    assert {s.content_hash for s in actual} == {s.content_hash for s in scenes}
    assert audit["counts"] == {"train": 1, "validation": 1}
    assert set(audit["references"].values()) == {str(a / "references"), str(b / "references")}
    assert merge_campaigns([a, b], output)["dataset_id"] == merged["dataset_id"]
    assert read_manifest(a / "accepted_manifest.json")[1][0].content_hash == scenes[0].content_hash
    status = a / "references" / "reference.json"
    data = json.loads(status.read_text())
    data["status"] = "nonconverged"
    status.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid accepted reference"):
        audit_campaign(output)


def test_sparse_limits():
    with pytest.raises(ValueError):
        SparseConfig(gap_pixels_min=1)
    with pytest.raises(ValueError):
        SparseConfig(epsilon_max=31)


def test_sparse_training_weight_targets_half_exposure_in_combined_corpus():
    records = [
        dict(scene=SimpleNamespace(family=f), budget=[n, n])
        for f in ("single_dielectric", "separated", "contact")
        for n in (48, 64, 96, 128)
    ]
    config = TrainingConfig(sparse_sample_weight=2)
    weights = np.array(training_sample_weights(records, config))
    # One sparse scene per two dense scenes becomes half of the sampling mass.
    assert weights[:4].sum() / weights.sum() == 0.5
    np.testing.assert_array_equal(training_sample_weights(records, TrainingConfig()), np.ones(12))
    # Optional budget balancing still composes with the family weights.
    balanced = training_sample_weights(records, replace(config, balance_budgets=True))
    np.testing.assert_allclose(balanced, weights / 3)
    for weight in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="training scalar"):
            TrainingConfig(sparse_sample_weight=weight)


def test_later_physics_selection_includes_sparse_families(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/start_physics_distillation_v6.py"
    loader = importlib.util.spec_from_file_location("sparse_physics_launcher", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    scenes = []
    for split, count in (("train", 16), ("validation", 4)):
        for family in (*module.FAMILIES, *FAMILIES):
            scenes.extend(
                SimpleNamespace(scene_id=f"{split}-{family}-{i:03d}", split=split, family=family)
                for i in range(count)
            )
    monkeypatch.setattr(module, "read_manifest", lambda _: ({}, scenes))
    metadata = tmp_path / "targets.json"
    metadata.write_text(
        json.dumps(
            dict(
                samples=[
                    dict(scene_id=s.scene_id, budget=[n, n]) for s in scenes for n in module.BUDGETS
                ]
            )
        )
    )
    selected, lanes = module._select_scenes("unused", metadata)
    assert set(selected["train"]) == set((*module.FAMILIES, *FAMILIES))
    assert sum(map(len, lanes)) == 160
    assert len({scene_id for lane in lanes for scene_id in lane}) == 160
