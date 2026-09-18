import hashlib
import json

import numpy as np
import pytest
import torch

from fdtdmesh.data.generate import GenerationConfig, make_scene
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.mesh import MESH_POLICY
from fdtdmesh.ml import ResUNet, load_model, save_model
from fdtdmesh.physics import (
    PHYSICS_TARGET_VERSION,
    PhysicsDataset,
    SearchConfig,
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
