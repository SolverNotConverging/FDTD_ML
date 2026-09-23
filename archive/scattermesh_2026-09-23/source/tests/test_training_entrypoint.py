import importlib.util
import json
from pathlib import Path


def load_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_mesh_distillation.py"
    spec = importlib.util.spec_from_file_location("mesh_distillation_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_run_requires_all_outputs_and_matching_provenance(tmp_path):
    trainer = load_trainer()
    provenance = {"dataset": "abc", "config": "def"}
    assert trainer.completed_run_matches(tmp_path, provenance) is False

    for name in ("checkpoint.pt", "history.json", "predicted_meshes.json"):
        (tmp_path / name).write_bytes(b"complete")
    summary = {
        "status": "complete",
        "source_hashes": provenance,
        "checkpoint_sha256": trainer.sha256_file(tmp_path / "checkpoint.pt"),
        "predicted_meshes_sha256": trainer.sha256_file(tmp_path / "predicted_meshes.json"),
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert trainer.completed_run_matches(tmp_path, provenance) is True
    assert trainer.completed_run_matches(tmp_path, {"dataset": "changed"}) is False

    (tmp_path / "checkpoint.pt").write_bytes(b"corrupt checkpoint")
    assert trainer.completed_run_matches(tmp_path, provenance) is False


def test_training_lock_can_be_acquired_and_released(tmp_path):
    trainer = load_trainer()
    handle = trainer.acquire_training_lock(tmp_path)
    try:
        assert (tmp_path / ".training.lock").is_file()
        assert handle.closed is False
    finally:
        handle.close()


def test_training_provenance_hashes_target_arrays(tmp_path):
    trainer = load_trainer()
    arrays = tmp_path / "targets.npz"
    arrays.write_bytes(b"first target tensor")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"arrays": arrays.name}))
    config = tmp_path / "config.json"
    config.write_text("{}")

    before = trainer.source_hashes(dataset, config)
    arrays.write_bytes(b"changed target tensor")
    after = trainer.source_hashes(dataset, config)

    assert before["dataset"] == after["dataset"]
    assert before["dataset_arrays"] != after["dataset_arrays"]
