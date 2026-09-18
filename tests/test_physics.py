import hashlib
import json

import numpy as np
import pytest
import torch

from fdtdmesh.data.generate import GenerationConfig, make_scene
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.evaluation import EvaluationConfig
from fdtdmesh.mesh import MESH_POLICY
from fdtdmesh.ml import ResUNet, load_model, save_model
from fdtdmesh.physics import (
    PHYSICS_TARGET_VERSION,
    PhysicsDataset,
    SearchConfig,
    _physics_evaluation_report,
    _reference,
    candidate_densities,
    load_physics_targets,
    pareto_front,
    score_candidates,
)


def small_scenes():
    config = GenerationConfig(
        min_objects=1, max_objects=1, raster_size=32, size_min=0.125, size_max=0.3
    )
    scenes = [
        make_scene(13, "train", config=config),
        make_scene(19, "validation", config=config),
    ]
    for scene in scenes:
        scene.budgets = [[32, 32]]
    return scenes


def test_candidate_families_are_positive_seeded_and_conditioned(tmp_path):
    scene = small_scenes()[0]
    path = tmp_path / "model.pt"
    torch.manual_seed(3)
    save_model(path, ResUNet(2), raster_shape=(32, 32))
    model, metadata = load_model(path)
    config = SearchConfig(perturbations=2, seed=7)
    a = candidate_densities(scene, [32, 32], model, metadata, config, "cpu")
    b = candidate_densities(scene, [32, 32], model, metadata, config, "cpu")
    assert [row["name"] for row in a] == [
        "uniform",
        "heuristic",
        "cnn",
        "uniform_heuristic_mix",
        "uniform_cnn_mix",
        "heuristic_cnn_mix",
        "cnn_perturb_0",
        "cnn_perturb_1",
    ]
    for left, right in zip(a, b):
        for x, y in zip(left["density"], right["density"]):
            assert len(x) == 32 and np.all(x > 0) and np.isclose(x.mean(), 1)
            np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[-1]["density"][0], a[-2]["density"][0])


def test_pareto_and_scalar_selection_keep_underlying_tradeoff():
    def row(name, error, updates, settled=True):
        return {
            "name": name,
            "status": "ok",
            "settled": settled,
            "metrics": {"waveform_l2_max": error, "spectrum_l2_max": error / 2},
            "diagnostics": {"cell_updates": updates},
        }

    rows = [
        row("uniform", 0.10, 100),
        row("heuristic", 0.05, 120),
        row("cheap", 0.08, 80),
        row("unsettled", 0.001, 50, settled=False),
        {"name": "failed", "status": "failed"},
    ]
    selected, front = score_candidates(rows, beta=0.02)
    assert selected["name"] == "heuristic"
    assert front == ["cheap", "heuristic"]
    assert pareto_front(rows) == front
    assert rows[0]["cost_ratio"] == 1 and rows[2]["cost_ratio"] == 0.8


@pytest.mark.parametrize(
    "options",
    [
        {"beta": -1},
        {"physics_weight": 1.1},
        {"perturbations": -1},
        {"perturbation_smoothing": 0},
        {"mixture_fraction": -0.1},
    ],
)
def test_invalid_search_config(options):
    with pytest.raises(ValueError):
        SearchConfig(**options)


def test_physics_target_integrity_and_dataset(tmp_path):
    scenes = small_scenes()
    manifest = tmp_path / "manifest.json"
    value = write_manifest(manifest, scenes, generation={"version": 3, "test": True})
    targets = tmp_path / "physics_targets.npz"
    x = np.full((2, 32), 1 / 32, dtype=np.float32)
    y = np.full((2, 32), 1 / 32, dtype=np.float32)
    np.savez_compressed(targets, target_x=x, target_y=y)
    metadata = {
        "schema_version": 1,
        "physics_target_version": PHYSICS_TARGET_VERSION,
        "dataset_id": value["dataset_id"],
        "targets_sha256": hashlib.sha256(targets.read_bytes()).hexdigest(),
        "mesh_policy": MESH_POLICY,
        "raster_shape": [32, 32],
        "samples": [
            {
                "scene_id": scene.scene_id,
                "scene_hash": scene.content_hash,
                "split": scene.split,
                "budget": [32, 32],
            }
            for scene in scenes
        ],
    }
    targets.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    _, _, loaded, tx, ty = load_physics_targets(manifest, targets)
    assert loaded["targets_sha256"] == metadata["targets_sha256"]
    np.testing.assert_array_equal(tx, x)
    np.testing.assert_array_equal(ty, y)
    dataset = PhysicsDataset(manifest, targets, "validation")
    assert len(dataset) == 1 and dataset[0]["raster"].shape == (9, 32, 32)
    assert dataset.target_id == metadata["targets_sha256"]
    np.savez_compressed(targets, target_x=2 * x, target_y=y)
    with pytest.raises(ValueError, match="do not match"):
        load_physics_targets(manifest, targets)


def test_external_reference_corpus_is_loaded_without_recomputation(tmp_path, monkeypatch):
    scene = small_scenes()[0]
    root = tmp_path / "references"
    directory = root / scene.scene_id
    directory.mkdir(parents=True)
    status = {
        "status": "converged",
        "accepted_budget": [64, 64],
        "scene_id": scene.scene_id,
        "scene_hash": scene.content_hash,
        "duration": scene.t_end,
    }
    (directory / "reference.json").write_text(json.dumps(status), encoding="utf-8")
    (directory / "evaluated_scene.json").write_text(json.dumps(scene.to_dict()), encoding="utf-8")
    times = np.linspace(0, scene.t_end, 17)
    frequencies = np.linspace(scene.f_min, scene.f_max, 5)
    waveforms = np.zeros((17, len(scene.receivers)))
    np.savez_compressed(
        directory / "reference_latest.npz",
        times=times,
        frequencies=frequencies,
        waveforms=waveforms,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("External references must not be recomputed")

    monkeypatch.setattr("fdtdmesh.physics.converge_reference", forbidden)
    loaded, effective, loaded_times, loaded_frequencies, loaded_waveforms = _reference(
        scene, tmp_path / "search", EvaluationConfig(), reference_root=root
    )
    assert loaded["status"] == "converged" and effective.to_dict() == scene.to_dict()
    np.testing.assert_array_equal(loaded_times, times)
    np.testing.assert_array_equal(loaded_frequencies, frequencies)
    np.testing.assert_array_equal(loaded_waveforms, waveforms)


def test_physics_evaluation_summary_uses_paired_heldout_errors():
    def row(strategy, error, updates):
        return {
            "scene_id": "test_iid-00000",
            "budget": [64, 64],
            "strategy": strategy,
            "status": "ok",
            "metrics": {"waveform_l2_max": error, "spectrum_l2_max": error / 2},
            "diagnostics": {"cell_updates": updates},
        }

    report = _physics_evaluation_report(
        {"dataset_id": "example"},
        [{"scene_id": "test_iid-00000", "status": "converged"}],
        [row("imitation", 0.2, 100), row("distilled", 0.1, 110)],
    )
    assert report["accepted_references"] == 1
    assert report["summary"]["distilled"]["median_em_error"] == pytest.approx(0.1)
    assert report["paired_vs_imitation"] == {"count": 1, "distilled_lower_error": 1}
