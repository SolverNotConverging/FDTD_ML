import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "scripts/continue_circle_remediation_pipeline.py"
    spec = importlib.util.spec_from_file_location("circle_remediation_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_snapshot_waits_for_retryable_unsettled_records(tmp_path):
    pipeline = load_pipeline()
    campaign = {
        "condition_ids": ["a", "b"],
        "candidate_names": ["uniform"],
        "duration_schedule_s": [1.0, 2.0],
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign))
    output = tmp_path / "output"
    for case_id, record in (
        ("a_uniform", {"status": "accepted", "config": {"duration_attempt_index": 0}}),
        ("b_uniform", {"status": "unsettled", "config": {"duration_attempt_index": 0}}),
    ):
        directory = output / "cases" / case_id
        directory.mkdir(parents=True)
        (directory / "record.json").write_text(json.dumps(record))

    snapshot = pipeline.campaign_snapshot(campaign_path, output)
    assert snapshot == {
        "planned": 2,
        "completed": 1,
        "remaining": 1,
        "malformed": 0,
        "status_counts": {"accepted": 1},
        "ready": False,
    }

    terminal = {"status": "unsettled", "config": {"duration_attempt_index": 1}}
    (output / "cases/b_uniform/record.json").write_text(json.dumps(terminal))
    snapshot = pipeline.campaign_snapshot(campaign_path, output)
    assert snapshot["completed"] == 2
    assert snapshot["status_counts"] == {"accepted": 1, "unsettled": 1}
    assert snapshot["ready"]
