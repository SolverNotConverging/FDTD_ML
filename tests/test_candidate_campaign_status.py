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
    assert result["remaining"] == 3
    assert result["simulation_status_counts"] == {"ok": 1}
    assert result["cache_status_counts"] == {"valid": 1, "malformed": 1, "missing": 2}
    assert result["recent_completions"] == 1
    assert result["recent_completion_rate_per_hour"] == 30.0
    assert result["estimated_remaining_seconds"] == 360.0
