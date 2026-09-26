import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from scattermesh import qualification_v2
from scattermesh.candidates_v2 import candidate_axes
from scattermesh.curriculum_v2 import (
    generate_c8_c9_targets,
    generate_compact_poc_scenes,
    generate_development_scenes,
    objects_from_scene,
)
from scattermesh.grid import Grid
from scattermesh.monitor_v2 import widest_non_pml_monitor_bounds
from scattermesh.profile_poc import (
    COMPACT_TEACHER_NAMES,
    DEVELOPMENT_BUDGETS,
    PROBE_ROLES,
    _estimate_campaign_hours,
    _raw_accuracy_headroom,
    _representative_costs,
    run_compact_profile,
)
from scattermesh.simulation_v2 import _failure_status

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_mesh_cnn_v2.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_mesh_cnn_v2", SCRIPT_PATH)
run_mesh_cnn_v2 = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(run_mesh_cnn_v2)


def test_compact_profile_records_gpu_unavailable_without_claiming_estimate(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    result = run_compact_profile(tmp_path, devices=())

    assert result["status"] == "gpu_unavailable"
    assert result["seed"] == 20260924
    assert result["estimated_compact_campaign_hours"] is None
    assert len(result["scenes"]) == 8
    assert json.loads((tmp_path / "compact_profile.json").read_text()) == result


def test_development_probe_roles_exist_and_accuracy_headroom_uses_raw_loss():
    scenes = generate_development_scenes()
    roles = {scene["development_role"] for scene in scenes}
    assert set(PROBE_ROLES) <= roles
    gain, best = _raw_accuracy_headroom(
        [
            {"name": "uniform", "accepted": True, "joint_scattering_loss": 0.1},
            {"name": "fast_but_less_accurate", "accepted": True, "joint_scattering_loss": 0.11},
            {"name": "interface", "accepted": True, "joint_scattering_loss": 0.08},
        ]
    )
    assert gain == 1.25
    assert best == "interface"


def test_compact_development_and_target_grids_have_legal_low_budget_monitors():
    cases = [
        (generate_development_scenes(), COMPACT_TEACHER_NAMES),
        (generate_c8_c9_targets(), ("uniform", "interface")),
    ]
    for scenes, policies in cases:
        for scene in scenes:
            bounds = [obj.bounds for obj in objects_from_scene(scene)]
            for policy in policies:
                x, y, _ = candidate_axes(scene, 32, policy)
                monitor = widest_non_pml_monitor_bounds(Grid(x, y, max_ratio=3), bounds, 0.12)
                assert monitor[0] < monitor[1] and monitor[2] < monitor[3]


def test_monitor_stencil_failure_is_classified_as_geometry_incompatible():
    assert (
        _failure_status(ValueError("Monitor interpolation stencil enters PML"))
        == "geometry_incompatible"
    )


def test_compact_cost_model_counts_references_and_separates_campaign_phases():
    development = generate_development_scenes()
    rows = []
    for scene in development:
        teachers = []
        for cells in DEVELOPMENT_BUDGETS:
            candidates = [
                {
                    "name": name,
                    "accepted": True,
                    "Nt": 100,
                    "wall_seconds": 0.5,
                }
                for name in COMPACT_TEACHER_NAMES
            ]
            teachers.append({"cells": cells, "candidates": candidates})
        rows.append(
            {
                "stage": scene["stage"],
                "teachers": teachers,
                "reference_observations": [{"status": "accepted", "wall_seconds": 10.0}],
            }
        )

    costs = _representative_costs(rows)
    projection = _estimate_campaign_hours(
        rows,
        generate_compact_poc_scenes(seed=20260924),
        costs,
        device_count=4,
        train_benchmark={"microbatch": 8, "step_seconds": 0.01},
        seed=20260924,
    )

    phases = projection["estimated_fdt_wall_hours_by_phase"]
    assert phases["data"] > 0
    assert phases["validation"] > 0
    assert phases["test"] > phases["validation"]
    assert projection["estimated_fdt_gpu_hours"] > sum(phases.values())
    assert projection["missing_cost_cells"] == []


def test_profile_compact_cli_uses_requested_output_and_forwards_options(tmp_path, monkeypatch):
    called = {}

    def fake_profile(output, **kwargs):
        called["output"] = output
        called.update(kwargs)
        return {"status": "gpu_unavailable"}

    monkeypatch.setattr(run_mesh_cnn_v2, "run_compact_profile", fake_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mesh_cnn_v2.py",
            "profile-compact",
            "--output",
            str(tmp_path),
            "--seed",
            "12",
            "--max-campaign-hours",
            "20",
        ],
    )

    run_mesh_cnn_v2.main()

    assert called["output"] == tmp_path / "profile"
    assert called["seed"] == 12
    assert called["max_campaign_hours"] == 20


def test_freeze_compact_cli_defaults_to_profile_compact_output(monkeypatch):
    called = {}

    def fake_freeze(profile_path, manifest_path, **kwargs):
        called["profile"] = profile_path
        called["manifest"] = manifest_path
        called.update(kwargs)
        return {"lineage_count": 104}

    monkeypatch.setattr(run_mesh_cnn_v2, "freeze_compact_manifest", fake_freeze)
    monkeypatch.setattr(sys, "argv", ["run_mesh_cnn_v2.py", "freeze-compact"])

    run_mesh_cnn_v2.main()

    default_root = run_mesh_cnn_v2.DEFAULT_COMPACT_OUTPUT
    assert called["profile"] == default_root / "profile" / "compact_profile.json"
    assert called["manifest"] == default_root / "manifest.json"
    assert called["seed"] == 20260924


def test_compact_launcher_stops_before_manifest_when_profile_gate_fails(tmp_path, monkeypatch):
    def fake_profile(profile_dir, *, seed, max_campaign_hours):
        return {
            "seed": seed,
            "gate": "gpu_unavailable",
            "status": "gpu_unavailable",
            "estimated_compact_campaign_hours": None,
        }

    monkeypatch.setattr(run_mesh_cnn_v2, "run_compact_profile", fake_profile)
    status = run_mesh_cnn_v2.launch_compact(tmp_path, seed=17)

    assert status["phase"] == "development_profile_stopped"
    assert status["reason"] == "gpu_unavailable"
    assert not (tmp_path / "manifest.json").exists()


def test_compact_reference_uses_frozen_successive_grid_policy_and_resumes(tmp_path, monkeypatch):
    scene = generate_development_scenes()[0]
    policy = {
        "protocol": "successive_uniform_compact_v1",
        "spatial_levels": [96, 128],
        "complex_tolerance": 0.01,
        "width_tolerance": 0.02,
        "reference_tail_ratio": 1e-5,
        "pml_thickness_m": 0.12,
        "material_samples": 12,
        "durations_s": [70e-9],
    }
    calls = []

    def fake_evaluate_case(target, cells, angle, name, directory, **kwargs):
        calls.append(cells)
        directory.mkdir(parents=True, exist_ok=True)
        field = np.ones((3, 180), dtype=np.complex128)
        np.savez(
            directory / "spectra.npz",
            complex_far_field=field,
            reference_complex_far_field=2 * field,
        )
        return {
            "status": "accepted",
            "accepted": True,
            "Nt": cells,
            "dt": 1e-12,
            "wall_seconds": 0.2,
            "tail_peak_over_global_peak": 1e-6,
        }, False

    monkeypatch.setattr(qualification_v2, "evaluate_case", fake_evaluate_case)
    report = qualification_v2.qualify_compact_reference(
        scene,
        0,
        0.0,
        tmp_path,
        policy,
        calibration_profile_sha256="profile-hash",
        device="cpu",
    )

    assert report["accepted"]
    assert report["selected_cells"] == 128
    assert report["reference_source"] == "analytic_circle"
    assert report["observations"][-1]["complex_change_from_previous"] == 0.0
    assert calls == [96, 128]
    cached = qualification_v2.qualify_compact_reference(
        scene,
        0,
        0.0,
        tmp_path,
        policy,
        calibration_profile_sha256="profile-hash",
        device="cpu",
    )
    assert cached == report
    assert calls == [96, 128]
