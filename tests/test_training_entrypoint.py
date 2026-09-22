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
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "complete", "source_hashes": provenance})
    )
    assert trainer.completed_run_matches(tmp_path, provenance) is True
    assert trainer.completed_run_matches(tmp_path, {"dataset": "changed"}) is False


def test_training_lock_can_be_acquired_and_released(tmp_path):
    trainer = load_trainer()
    handle = trainer.acquire_training_lock(tmp_path)
    try:
        assert (tmp_path / ".training.lock").is_file()
        assert handle.closed is False
    finally:
        handle.close()
