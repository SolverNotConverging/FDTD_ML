"""Adaptive reference escalation and hard-limit tests."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_reference_convergence_policy.py"
SPEC = importlib.util.spec_from_file_location("run_reference_convergence_policy", SCRIPT)
policy_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_module)


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
a=p.parse_args(); contour=[.23,.97,.23,.97] if a.larger_contour else [.3,.9,.3,.9]
config={'scene':a.scene,'cells':a.cells,'duration_ns':a.duration_ns,'device':a.device,
        'samples':a.samples,'pml':a.pml,'contour':contour,
        'source_sha256':json.loads(os.environ['SCATTERMESH_SOURCE_SHA256'])}
fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
failed=a.scene == 'eps30_small'
gates={name:{'measured':1.0 if failed else 0.0,'threshold':.5,'passed':not failed}
       for name in ('max_complex_relative_l2','max_phase_rms_degrees','tail_peak_over_global_peak')}
gates['max_analytic_series_relative_l2']={'measured':0.0,'threshold':.5,'passed':True}
record={'fingerprint':fingerprint,'status':'fail' if failed else 'pass','config':config,
        'finite_fields':True,'gates':gates}
out=a.output_root/a.scene/fingerprint[:12]; out.mkdir(parents=True,exist_ok=True)
(out/'record.json').write_text(json.dumps(record))
angles=np.linspace(0,2*np.pi,180,endpoint=False); field=np.ones((3,180),dtype=complex)
np.savez(out/'spectra.npz',complex_numerical=field,complex_analytic=field,
         frequencies=np.array([.8e9,1e9,1.2e9]),angles=angles)
"""
    )


def test_policy_accepts_independent_probes_and_skips_at_hard_limit(tmp_path):
    runner = tmp_path / "fake_runner.py"
    fake_runner(runner)
    policy_path = tmp_path / "policy.json"
    raw = dict(
        schema_version=1,
        campaign_id="adaptive_test",
        output_root=str(tmp_path / "references"),
        variation_threshold=0.005,
        scenes=[
            dict(
                scene_id="accepted",
                scene="eps30_lossy",
                cell_levels=[32, 48],
                duration_levels_ns=[10, 20],
                sample_levels=[2, 4],
            ),
            dict(
                scene_id="hard_limit",
                scene="eps30_small",
                cell_levels=[32, 48],
                duration_levels_ns=[10, 20],
                sample_levels=[2, 4],
            ),
        ],
    )
    policy_path.write_text(json.dumps(raw))
    policy = policy_module.validate_policy(raw, policy_path)
    output = tmp_path / "campaign"
    state = policy_module.run_policy(
        policy, output, ["cuda:0", "cuda:1"], runner=runner, max_attempts=1
    )
    assert state["status"] == "complete"
    assert state["wave_count"] == 2
    assert len(state["attempts"]) == 7
    accepted = state["scenes"]["accepted"]
    assert accepted["decision"] == "accepted"
    assert set(accepted["probes"]) == {"spatial", "duration", "quadrature", "contour"}
    assert all(value["maximum"] == 0 for value in accepted["complex_field_variations"].values())
    skipped = state["scenes"]["hard_limit"]
    assert skipped["decision"] == "skipped"
    assert skipped["reason"] == "hard_limit"
    assert skipped["hard_limit_reason"] == "individual_gate_failure"

    resumed = policy_module.run_policy(
        policy, output, ["cuda:0", "cuda:1"], runner=runner, max_attempts=1
    )
    assert resumed["wave_count"] == 2
    assert len(resumed["attempts"]) == 7

    accepted_record = Path(resumed["scenes"]["accepted"]["accepted_record"])
    accepted_record.with_name("spectra.npz").write_bytes(b"corrupt")
    repaired = policy_module.run_policy(
        policy, output, ["cuda:0", "cuda:1"], runner=runner, max_attempts=1
    )
    assert repaired["status"] == "complete"
    assert repaired["wave_count"] == 3
    assert repaired["scenes"]["accepted"]["decision"] == "accepted"
    base_key = policy_module.attempt_key(
        "accepted", repaired["scenes"]["accepted"]["accepted_config"]
    )
    assert repaired["attempts"][base_key]["invalidations"] == 1
    assert repaired["attempts"][base_key]["total_process_attempts"] == 2
