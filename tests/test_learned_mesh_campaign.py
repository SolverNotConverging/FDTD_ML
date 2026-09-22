import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hash_shards_are_deterministic_and_cover_every_case_once():
    runner = load_script("run_learned_mesh_pilot.py")
    case_ids = [f"sample_{index}_cnn" for index in range(100)]
    assignments = {
        case_id: runner.stable_shard(case_id, 4) for case_id in case_ids
    }

    assert assignments == {
        case_id: runner.stable_shard(case_id, 4) for case_id in case_ids
    }
    assert set(assignments.values()) == {0, 1, 2, 3}
    assert sum(
        runner.stable_shard(case_id, 4) == shard
        for shard in range(4)
        for case_id in case_ids
    ) == len(case_ids)


def test_training_snapshot_requires_complete_summary_dataset_and_checkpoint(tmp_path):
    pipeline = load_script("continue_learned_mesh_evaluation.py")
    dataset = tmp_path / "dataset.json"
    training = tmp_path / "training"
    training.mkdir()

    assert pipeline.training_snapshot(dataset, training)["ready"] is False
    dataset.write_text("{}")
    (training / "summary.json").write_text(
        json.dumps({"status": "running", "epochs_completed": 4})
    )
    (training / "checkpoint.pt").write_bytes(b"checkpoint")
    assert pipeline.training_snapshot(dataset, training)["ready"] is False
    (training / "summary.json").write_text(
        json.dumps({"status": "complete", "epochs_completed": 10, "best_epoch": 8})
    )

    snapshot = pipeline.training_snapshot(dataset, training)

    assert snapshot["ready"] is True
    assert snapshot["training_status"] == "complete"
    assert snapshot["epochs_completed"] == 10
    assert snapshot["best_epoch"] == 8


def test_evaluation_snapshot_counts_only_valid_expected_records(tmp_path):
    pipeline = load_script("continue_learned_mesh_evaluation.py")
    output = tmp_path / "output"
    accepted = output / "cases/a/record.json"
    unsettled = output / "cases/b/record.json"
    malformed = output / "cases/c/record.json"
    for path in (accepted, unsettled, malformed):
        path.parent.mkdir(parents=True)
    accepted.write_text(json.dumps({"status": "accepted"}))
    unsettled.write_text(json.dumps({"status": "unsettled"}))
    malformed.write_text("{")

    snapshot = pipeline.evaluation_snapshot(["a", "b", "c", "d"], output)

    assert snapshot == {
        "planned": 4,
        "completed": 2,
        "accepted": 1,
        "unsettled": 1,
        "malformed": 1,
        "remaining": 2,
    }


def test_expected_case_ids_include_only_frozen_heldout_splits(tmp_path):
    pipeline = load_script("continue_learned_mesh_evaluation.py")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "examples": [
                    {"sample_id": "train", "split": "train"},
                    {"sample_id": "validation", "split": "validation"},
                    {"sample_id": "test", "split": "test"},
                ]
            }
        )
    )

    assert pipeline.expected_case_ids(dataset) == ["validation_cnn", "test_cnn"]


def test_summary_can_name_the_frozen_physics_gate(tmp_path):
    runner = load_script("run_learned_mesh_pilot.py")
    output = tmp_path / "evaluation"
    candidates = tmp_path / "candidates"
    cases = []
    for split in ("validation", "test"):
        sample_id = f"{split}_sample"
        case_id = f"{sample_id}_cnn"
        learned = output / "cases" / case_id / "record.json"
        uniform = candidates / "cases" / f"{sample_id}_uniform" / "record.json"
        teacher = candidates / "cases" / f"{sample_id}_region" / "record.json"
        for path, payload in (
            (
                learned,
                {
                    "accepted": True,
                    "joint_scattering_loss": 0.1,
                    "complex_mse_loss": 0.08,
                    "rcs_log_loss": 0.08,
                    "Nt": 100,
                },
            ),
            (
                uniform,
                {
                    "joint_scattering_loss": 0.2,
                    "complex_mse_loss": 0.16,
                    "rcs_log_loss": 0.16,
                    "Nt": 100,
                },
            ),
            (
                teacher,
                {
                    "joint_scattering_loss": 0.08,
                    "complex_mse_loss": 0.064,
                    "rcs_log_loss": 0.064,
                    "Nt": 100,
                },
            ),
        ):
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload))
        cases.append(
            {
                "case_id": case_id,
                "sample_id": sample_id,
                "split": split,
                "example": {
                    "cells_x": 48,
                    "epsilon_r": 20 if split == "validation" else 5,
                    "sigma_e_s_per_m": 0.1,
                    "best_candidate": "region",
                },
            }
        )

    runner.summarize(
        cases,
        output,
        candidates,
        exponent=0.1,
        success_decision="passes_frozen_physics_evaluation",
    )
    report = json.loads((output / "report.json").read_text())

    assert report["decision"] == "passes_frozen_physics_evaluation"
    assert report["splits"]["validation"]["p10_improvement_over_uniform"] == 2.0
    assert report["splits"]["test"]["p90_score_ratio_to_teacher"] == 1.25
    assert report["by_budget"]["48"]["median_complex_improvement_over_uniform"] == 2.0
    assert report["by_contrast_tier"]["epsilon_r_gt_10"][
        "median_rcs_improvement_over_uniform"
    ] == 2.0
    assert report["by_split_and_contrast_tier"]["test"]["epsilon_r_le_10"][
        "case_count"
    ] == 1
    assert (output / "evaluation_summary.png").is_file()
