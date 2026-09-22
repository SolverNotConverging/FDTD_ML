"""Restart, assignment, and terminal-outcome tests for the GPU campaign scheduler."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_multi_gpu_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_multi_gpu_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def raw_plan(output_root):
    return dict(
        schema_version=1,
        campaign_id="scheduler_test",
        output_root=str(output_root),
        tasks=[
            dict(task_id="pass", scene="eps12_small", cells=32, duration_ns=1),
            dict(task_id="nonconverged", scene="eps30_lossy", cells=32, duration_ns=1),
            dict(task_id="process_failure", scene="eps30_small", cells=32, duration_ns=1),
        ],
    )


def fake_runner(path):
    path.write_text(
        """#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--scene'); p.add_argument('--cells', type=int)
p.add_argument('--duration-ns', type=float); p.add_argument('--device')
p.add_argument('--samples', type=int); p.add_argument('--pml', type=float)
p.add_argument('--output-root', type=Path); p.add_argument('--larger-contour', action='store_true')
a=p.parse_args()
if a.scene == 'eps30_small':
    raise SystemExit(4)
contour=[.23,.97,.23,.97] if a.larger_contour else [.3,.9,.3,.9]
config={'scene': a.scene, 'cells': a.cells, 'duration_ns': a.duration_ns,
        'device': a.device, 'samples': a.samples, 'pml': a.pml, 'contour': contour,
        'source_sha256': json.loads(os.environ['SCATTERMESH_SOURCE_SHA256'])}
fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
record={'fingerprint': fingerprint, 'status': 'fail' if a.scene == 'eps30_lossy' else 'pass',
        'config': config}
out=a.output_root/a.scene/(a.scene+'-attempt'); out.mkdir(parents=True, exist_ok=True)
(out/'record.json').write_text(json.dumps(record))
angles=np.linspace(0,2*np.pi,180,endpoint=False); field=np.ones((3,180),dtype=complex)
np.savez(out/'spectra.npz', complex_numerical=field, complex_analytic=field,
         frequencies=np.array([.8e9,1e9,1.2e9]), angles=angles)
"""
    )


def test_scheduler_retains_results_retries_process_failures_and_resumes(tmp_path):
    runner = tmp_path / "fake_runner.py"
    fake_runner(runner)
    plan_path = tmp_path / "plan.json"
    raw = raw_plan(tmp_path / "references")
    plan_path.write_text(json.dumps(raw))
    plan = campaign.validate_plan(raw, plan_path, ["cuda:0", "cuda:1"])
    assert [task["device"] for task in plan["tasks"]] == ["cuda:0", "cuda:1", "cuda:0"]

    output = tmp_path / "campaign"
    state = campaign.run_campaign(plan, output, runner=runner, max_attempts=2)
    assert state["status"] == "complete_with_failures"
    assert state["tasks"]["pass"]["status"] == "succeeded"
    assert state["tasks"]["pass"]["attempts"] == 1
    assert state["tasks"]["nonconverged"]["status"] == "nonconverged"
    assert state["tasks"]["nonconverged"]["attempts"] == 1
    assert state["tasks"]["process_failure"]["status"] == "failed"
    assert state["tasks"]["process_failure"]["attempts"] == 2
    assert len(state["tasks"]["process_failure"]["process_failures"]) == 2

    resumed = campaign.run_campaign(plan, output, runner=runner, max_attempts=2)
    assert resumed["tasks"]["pass"]["attempts"] == 1
    assert resumed["tasks"]["nonconverged"]["attempts"] == 1
    assert resumed["tasks"]["process_failure"]["attempts"] == 2
    assert json.loads((output / "summary.json").read_text())["counts"] == {
        "failed": 1,
        "nonconverged": 1,
        "succeeded": 1,
    }

    pass_result = Path(resumed["tasks"]["pass"]["result_path"])
    pass_result.with_name("spectra.npz").write_bytes(b"corrupt")
    repaired = campaign.run_campaign(plan, output, runner=runner, max_attempts=2)
    assert repaired["tasks"]["pass"]["status"] == "succeeded"
    assert repaired["tasks"]["pass"]["attempts"] == 2


def test_plan_validation_rejects_duplicate_ids_and_unavailable_devices(tmp_path):
    plan_path = tmp_path / "plan.json"
    raw = raw_plan(tmp_path / "references")
    raw["tasks"][1]["task_id"] = "pass"
    with pytest.raises(ValueError, match="Duplicate"):
        campaign.validate_plan(raw, plan_path, ["cuda:0"])

    raw = raw_plan(tmp_path / "references")
    raw["tasks"][0]["device"] = "cuda:3"
    with pytest.raises(ValueError, match="unavailable"):
        campaign.validate_plan(raw, plan_path, ["cuda:0"])
