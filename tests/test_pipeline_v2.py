"""Target eligibility, exact grids and accuracy-led pilot contracts."""

import json

import numpy as np
import pytest

from scattermesh.campaign_v2 import (
    freeze_compact_manifest,
    freeze_manifest,
    mark_unfinished,
    scoring_fingerprint,
)
from scattermesh.candidates_v2 import CANDIDATE_NAMES, candidate_axes
from scattermesh.curriculum_v2 import generate_c8_c9_targets, generate_lineages
from scattermesh.dataset_v2 import build_new_dataset
from scattermesh.scoring_v2 import penalty_sensitivity, rank_candidates, soft_dt_score
from scattermesh.simulation_v2 import numerical_source_hashes


def test_weak_dt_penalty_keeps_raw_accuracy_primary():
    rows = [
        {"name": "uniform", "accepted": True, "joint_scattering_loss": 0.1, "dt": 1.0},
        {"name": "focused", "accepted": True, "joint_scattering_loss": 0.07, "dt": 0.5},
    ]
    assert rank_candidates(rows)[0]["name"] == "focused"
    assert np.isclose(soft_dt_score(0.07, 0.5, 1.0), 0.07 * 2**0.05)
    assert list(penalty_sensitivity(rows)) == ["0.0", "0.02", "0.05", "0.1"]


def test_lineages_are_grouped_and_candidate_axes_are_legal():
    scenes = generate_lineages(128)
    assert len({scene["lineage_id"] for scene in scenes}) == 128
    assert {scene["split"] for scene in scenes} == {"train", "validation", "test"}
    assert sum(scene["material"]["kind"] == "pec" for scene in scenes) == 32
    scene = next(scene for scene in scenes if scene["family"] == "concave_polygon")
    for cells in (32, 48, 64, 96):
        x, y, repairs = candidate_axes(scene, cells, "interface")
        assert len(x) == len(y) == cells + 1
        assert x[0] == y[0] == 0
        assert np.isclose(x[-1], 1.2) and np.isclose(y[-1], 1.2)
        for axis in (x, y):
            widths = np.diff(axis)
            assert np.all(widths > 0)
            assert (
                max(np.max(widths[1:] / widths[:-1]), np.max(widths[:-1] / widths[1:])) <= 3 + 1e-9
            )
        assert all(0 <= repair <= 1 for repair in repairs)


def test_freeze_requires_measured_admissible_profile(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"gate": "report_before_bulk", "selected_lineages": None}))
    with pytest.raises(ValueError, match="does not admit"):
        freeze_manifest(profile, tmp_path / "manifest.json")


def test_compact_manifest_requires_cost_gate_and_keeps_test_scenes_frozen(tmp_path):
    profile = tmp_path / "compact_profile.json"
    reference_policy = {
        "protocol": "successive_uniform_compact_v1",
        "spatial_levels": [96, 128, 192, 256],
        "complex_tolerance": 0.01,
        "width_tolerance": 0.02,
        "reference_tail_ratio": 1e-5,
        "pml_thickness_m": 0.12,
        "material_samples": 12,
        "durations_s": [7e-8, 1.4e-7, 5.6e-7],
    }
    profile.write_text(
        json.dumps(
            {
                "gate": "ready_for_compact_campaign",
                "seed": 7,
                "estimated_compact_campaign_hours": 20.0,
                "teacher_search_calibration": {
                    "method": "96_evaluation_smooth_density_differential_evolution",
                    "estimated_teacher_search_wall_hours": 1.0,
                },
                "reference_policy": reference_policy,
                "reference_probes_pass": True,
                "positive_development_teacher_gain": True,
                "qualified_reference_count": 8,
                "missing_cost_cells": [],
                "numerical_source_hashes": numerical_source_hashes(),
                "scoring_fingerprint": scoring_fingerprint(),
            }
        )
    )
    path = tmp_path / "compact_manifest.json"
    manifest = freeze_compact_manifest(profile, path, seed=7)
    assert manifest == freeze_compact_manifest(profile, path, seed=7)
    assert manifest["split_counts"] == {"train": 64, "validation": 16, "test": 24}
    assert len(manifest["candidate_names"]) == 9
    assert manifest["teacher_search"]["evaluations_per_condition"] == 96
    assert manifest["reference_policy"] == reference_policy
    assert manifest["reference_policy_source_profile_sha256"]
    assert manifest["scoring_fingerprint"] == scoring_fingerprint()
    assert manifest["scoring_fingerprint"]["time_step_exponent"] == 0.05
    assert manifest["scoring_fingerprint"]["scattering_width_weight"] == 0.25
    assert len(manifest["evaluation_conditions"]["test"]) == 24
    assert all(
        scene["split"] == "test"
        for scene in manifest["scenes"]
        if scene["lineage_id"] in manifest["evaluation_conditions"]["test"]
    )
    profile.write_text(
        json.dumps(
            {
                "gate": "ready_for_compact_campaign",
                "seed": 7,
                "estimated_compact_campaign_hours": 25.0,
                "reference_policy": reference_policy,
                "reference_probes_pass": True,
                "positive_development_teacher_gain": True,
                "qualified_reference_count": 8,
                "missing_cost_cells": [],
                "numerical_source_hashes": numerical_source_hashes(),
                "scoring_fingerprint": scoring_fingerprint(),
            }
        )
    )
    with pytest.raises(ValueError, match="at most 24 hours"):
        freeze_compact_manifest(profile, tmp_path / "too_large.json", seed=7)


def test_incomplete_candidate_cannot_become_teacher(tmp_path):
    scene = generate_lineages(128)[0]
    manifest = {
        "scenes": [scene],
        "incidence_angles": [0.0],
        "budgets": [32],
        "frequencies_hz": [0.8e9, 1e9, 1.2e9],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    campaign = tmp_path / "campaign"
    qualification = campaign / "qualifications" / f"{scene['lineage_id']}_a0.json"
    qualification.parent.mkdir(parents=True)
    qualification.write_text(json.dumps({"accepted": True}))
    for name in CANDIDATE_NAMES:
        directory = campaign / "cases" / f"{scene['lineage_id']}_a0_n32_{name}"
        directory.mkdir(parents=True)
        accepted = name == "uniform"
        (directory / "record.json").write_text(
            json.dumps(
                {
                    "status": "accepted" if accepted else "incomplete",
                    "accepted": accepted,
                    "joint_scattering_loss": 0.1 if accepted else None,
                    "dt": 1e-11 if accepted else None,
                }
            )
        )
        if accepted:
            axis = np.linspace(0, 1.2, 33)
            np.savez(directory / "spectra.npz", x=axis, y=axis)
    with pytest.raises(ValueError, match="No reference-qualified"):
        build_new_dataset(manifest_path, campaign, tmp_path / "dataset")


def test_new_dataset_uses_manifest_candidates_and_excludes_c8_test_labels(tmp_path):
    train = generate_lineages(128)[0]
    train["split"] = "train"
    test = generate_c8_c9_targets(silhouette_count=3, distributed_count=3)[0]
    manifest = {
        "scenes": [train, test],
        "incidence_angles": [0.0],
        "budgets": [32],
        "frequencies_hz": [0.8e9, 1.0e9, 1.2e9],
        "candidate_names": ["uniform", "interface"],
        "conditions_by_scene": {train["lineage_id"]: [{"angle_index": 0, "cells": 32}]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    campaign = tmp_path / "campaign"
    qualification = campaign / "qualifications" / f"{train['lineage_id']}_a0.json"
    qualification.parent.mkdir(parents=True)
    qualification.write_text(json.dumps({"accepted": True}))
    axis = np.linspace(0, 1.2, 33)
    for index, name in enumerate(manifest["candidate_names"]):
        directory = campaign / "cases" / f"{train['lineage_id']}_a0_n32_{name}"
        directory.mkdir(parents=True)
        (directory / "record.json").write_text(
            json.dumps(
                {
                    "status": "accepted",
                    "accepted": True,
                    "joint_scattering_loss": 0.1 - index * 0.01,
                    "dt": 1e-11,
                }
            )
        )
        np.savez(directory / "spectra.npz", x=axis, y=axis)
    path = build_new_dataset(manifest_path, campaign, tmp_path / "dataset")
    metadata = json.loads(path.read_text())
    with np.load(path.parent / metadata["arrays"]) as arrays:
        assert arrays["candidate_mask"].shape == (1, 2)
        assert arrays["candidate_mask"].all()
    assert len(metadata["examples"]) == 1
    assert metadata["examples"][0]["split"] == "train"
    assert metadata["examples"][0]["scene"]["stage"] != "C8"


def test_deadline_creates_resumable_incomplete_records(tmp_path):
    scene = generate_lineages(128)[0]
    manifest = {"scenes": [scene], "incidence_angles": [0.0], "budgets": [32]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    counts = mark_unfinished(path, tmp_path / "campaign")
    assert counts == {"references": 1, "cases": len(CANDIDATE_NAMES)}
    assert mark_unfinished(path, tmp_path / "campaign") == {"references": 0, "cases": 0}
