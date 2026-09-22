import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_status():
    path = ROOT / "scripts/candidate_campaign_status.py"
    spec = importlib.util.spec_from_file_location("candidate_campaign_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_counts_expected_records_and_recent_rate(tmp_path):
    status = load_status()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"condition_ids": ["a", "b"], "candidate_names": ["u", "v"]}))
    cases = tmp_path / "out" / "cases"
    (cases / "a_u").mkdir(parents=True)
    (cases / "a_u" / "record.json").write_text(json.dumps({"status": "ok"}))
    (cases / "b_u").mkdir()
    (cases / "b_u" / "record.json").write_text("{")
    now = 2_000_000.0
    os.utime(cases / "a_u" / "record.json", (now - 120, now - 120))
    result = status.snapshot(campaign, tmp_path / "out", window_minutes=10, now=now)
    assert result["planned"] == 4
    assert result["completed"] == 1
    assert result["observed_records"] == 1
    assert result["remaining"] == 3
    assert result["simulation_status_counts"] == {"ok": 1}
    assert result["retrying_status_counts"] == {}
    assert result["cache_status_counts"] == {"valid": 1, "malformed": 1, "missing": 2}
    assert result["recent_completions"] == 1
    assert result["recent_completion_rate_per_hour"] == 30.0
    assert result["estimated_remaining_seconds"] == 360.0


def test_snapshot_reports_lineage_progress(tmp_path):
    status = load_status()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "dataset_id": "dataset",
                "condition_ids": ["a", "b"],
                "candidate_names": ["u", "v"],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "dataset",
                "geometries": [
                    {"geometry_id": "ga", "epsilon_r": 2.0},
                    {"geometry_id": "gb", "epsilon_r": 4.5},
                ],
                "conditions": [
                    {
                        "task_id": "a",
                        "geometry_id": "ga",
                        "lineage_id": "low",
                        "split": "train",
                    },
                    {
                        "task_id": "b",
                        "geometry_id": "gb",
                        "lineage_id": "high",
                        "split": "validation",
                    },
                ],
            }
        )
    )
    record = tmp_path / "out/cases/a_u/record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"status": "accepted"}))

    result = status.snapshot(campaign, tmp_path / "out", manifest=manifest)
    assert result["lineage_progress"] == {
        "high": {
            "split": "validation",
            "epsilon_r": [4.5],
            "planned": 2,
            "completed": 0,
            "remaining": 2,
            "simulation_status_counts": {},
        },
        "low": {
            "split": "train",
            "epsilon_r": [2.0],
            "planned": 2,
            "completed": 1,
            "remaining": 1,
            "simulation_status_counts": {"accepted": 1},
        },
    }


def test_snapshot_does_not_count_intermediate_unsettled_record_as_terminal(tmp_path):
    status = load_status()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "condition_ids": ["a", "b"],
                "candidate_names": ["u"],
                "duration_schedule_s": [1.0, 2.0, 4.0],
            }
        )
    )
    cases = tmp_path / "out" / "cases"
    for case_id, attempt_index in (("a_u", 1), ("b_u", 2)):
        path = cases / case_id / "record.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "status": "unsettled",
                    "config": {"duration_attempt_index": attempt_index},
                }
            )
        )

    result = status.snapshot(campaign, tmp_path / "out")

    assert result["observed_records"] == 2
    assert result["completed"] == 1
    assert result["remaining"] == 1
    assert result["simulation_status_counts"] == {"unsettled": 1}
    assert result["retrying_status_counts"] == {"unsettled": 1}
    assert result["cache_status_counts"] == {"valid": 2, "malformed": 0, "missing": 0}
