import json

import numpy as np
import pytest
import torch

from fdtdmesh.data.generate import GenerationConfig, make_scene
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.ml import ResUNet, load_model, save_model
from fdtdmesh.training import (
    TeacherDataset,
    TrainingConfig,
    axis_probability,
    build_teacher_targets,
    imitation_loss,
    load_teacher_targets,
    train_model,
)


def teacher_fixture(tmp_path):
    generation = GenerationConfig(
        min_objects=1,
        max_objects=1,
        raster_size=32,
        size_min=0.125,
        size_max=0.3,
    )
    scenes = [
        make_scene(13, "train", config=generation),
        make_scene(19, "validation", config=generation),
    ]
    for scene in scenes:
        scene.budgets = [[32, 32]]
        scene.raster_shape = [32, 32]
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, scenes, generation={"version": 3, "test": True})
    targets = tmp_path / "targets.npz"
    build_teacher_targets(manifest, targets)
    return manifest, targets


def test_axis_probability_and_imitation_gradients():
    density_x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], requires_grad=True)
    density_y = torch.tensor([[8.0, 4.0, 2.0, 1.0]], requires_grad=True)
    collars = torch.tensor([[0.125, 0.125]])
    probability = axis_probability(density_x, collars[:, 0])
    torch.testing.assert_close(probability.sum(1), torch.ones(1))
    assert probability[0, 0] < probability[0, -1]
    target_x = torch.full((1, 4), 0.25)
    target_y = torch.full((1, 4), 0.25)
    loss, lx, ly = imitation_loss(density_x, density_y, target_x, target_y, collars)
    torch.testing.assert_close(loss, (lx + ly) / 2)
    loss.backward()
    assert torch.isfinite(density_x.grad).all() and density_x.grad.abs().sum() > 0
    assert torch.isfinite(density_y.grad).all() and density_y.grad.abs().sum() > 0


def test_teacher_targets_are_legal_seeded_and_bound_to_manifest(tmp_path):
    manifest, targets = teacher_fixture(tmp_path)
    metadata = json.loads(targets.with_suffix(".json").read_text())
    assert metadata["teacher_version"] == 1 and len(metadata["samples"]) == 2
    _, _, _, x, y = load_teacher_targets(manifest, targets)
    np.testing.assert_allclose(x.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(y.sum(1), 1, atol=1e-6)
    assert TeacherDataset(manifest, targets, "train")[0]["raster"].shape == (9, 32, 32)
    with pytest.raises(ValueError, match="already exists"):
        build_teacher_targets(manifest, targets)
    arrays = np.load(targets)
    np.savez_compressed(targets, target_x=arrays["target_x"] * 2, target_y=arrays["target_y"])
    with pytest.raises(ValueError, match="do not match"):
        load_teacher_targets(manifest, targets)


def test_short_training_checkpoint_and_resume_contract(tmp_path):
    manifest, targets = teacher_fixture(tmp_path)
    output = tmp_path / "training"
    config = TrainingConfig(
        epochs=1,
        batch_size=1,
        width=2,
        repair_weight=0,
        projection_samples=1,
        seed=4,
    )
    report = train_model(manifest, targets, output, config=config, device="cpu", allow_dirty=True)
    assert len(report["history"]) == 1 and report["best_validation_loss"] >= 0
    model, metadata = load_model(output / "best.pt")
    assert isinstance(model, ResUNet)
    assert metadata["dataset_version"] == report["dataset_id"]
    assert metadata["training"]["epoch"] == 1
    resumed = TrainingConfig(**{**config.__dict__, "epochs": 2})
    report = train_model(
        manifest,
        targets,
        output,
        config=resumed,
        device="cpu",
        resume=output / "resume.pt",
        allow_dirty=True,
    )
    assert [row["epoch"] for row in report["history"]] == [1, 2]
    changed = TrainingConfig(**{**resumed.__dict__, "width": 3})
    with pytest.raises(ValueError, match="configuration"):
        train_model(
            manifest,
            targets,
            output,
            config=changed,
            device="cpu",
            resume=output / "resume.pt",
            allow_dirty=True,
        )


def test_checkpoint_training_metadata_validation(tmp_path):
    path = tmp_path / "model.pt"
    save_model(path, ResUNet(2), training_metadata={"epoch": 1})
    _, metadata = load_model(path)
    assert metadata["training"] == {"epoch": 1}
    with pytest.raises(ValueError, match="dictionary"):
        save_model(path, ResUNet(2), training_metadata="bad")
