import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "scripts/continue_exact_mesh_pipeline.py"
    spec = importlib.util.spec_from_file_location("continue_exact_mesh_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_record(root, case_id, payload):
    path = root / "cases" / case_id / "record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))


def test_campaign_snapshot_counts_terminal_rejections_without_a_retry_schedule(tmp_path):
    pipeline = load_pipeline()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps({"condition_ids": ["a", "b"], "candidate_names": ["u", "v"]})
    )
    output = tmp_path / "output"
    write_record(output, "a_u", {"status": "accepted"})
    write_record(output, "a_v", {"status": "unsettled"})
    write_record(output, "b_u", "{")

    snapshot = pipeline.campaign_snapshot(campaign, output)

    assert snapshot == {
        "planned": 4,
        "completed": 2,
        "remaining": 2,
        "malformed": 1,
        "status_counts": {"accepted": 1, "unsettled": 1},
        "retrying_counts": {},
        "ready": False,
    }


def test_campaign_snapshot_is_ready_only_after_all_accept(tmp_path):
    pipeline = load_pipeline()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps({"condition_ids": ["a", "b"], "candidate_names": ["u", "v"]})
    )
    output = tmp_path / "output"
    for case_id in ("a_u", "a_v", "b_u", "b_v"):
        write_record(output, case_id, {"status": "accepted"})

    snapshot = pipeline.campaign_snapshot(campaign, output)

    assert snapshot["ready"] is True
    assert snapshot["completed"] == snapshot["planned"] == 4
    assert snapshot["remaining"] == 0
    assert snapshot["status_counts"] == {"accepted": 4}
    assert snapshot["retrying_counts"] == {}


def test_campaign_snapshot_distinguishes_retryable_and_terminal_unsettled(tmp_path):
    pipeline = load_pipeline()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "condition_ids": ["a"],
                "candidate_names": ["uniform", "region"],
                "duration_schedule_s": [1.0, 2.0, 4.0],
            }
        )
    )
    output = tmp_path / "output"
    write_record(
        output,
        "a_uniform",
        {"status": "accepted", "config": {"duration_attempt_index": 0}},
    )
    write_record(
        output,
        "a_region",
        {"status": "unsettled", "config": {"duration_attempt_index": 1}},
    )

    retrying = pipeline.campaign_snapshot(campaign, output)
    assert retrying["completed"] == 1
    assert retrying["remaining"] == 1
    assert retrying["status_counts"] == {"accepted": 1}
    assert retrying["retrying_counts"] == {"unsettled": 1}
    assert retrying["ready"] is False

    write_record(
        output,
        "a_region",
        {"status": "unsettled", "config": {"duration_attempt_index": 2}},
    )
    terminal = pipeline.campaign_snapshot(campaign, output)
    assert terminal["completed"] == terminal["planned"] == 2
    assert terminal["remaining"] == 0
    assert terminal["status_counts"] == {"accepted": 1, "unsettled": 1}
    assert terminal["retrying_counts"] == {}
    assert terminal["ready"] is True
