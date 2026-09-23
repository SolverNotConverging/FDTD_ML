"""Target eligibility, exact grids and accuracy-led pilot contracts."""

import json

import numpy as np
import pytest

from scattermesh.campaign_v2 import freeze_manifest, mark_unfinished
from scattermesh.candidates_v2 import CANDIDATE_NAMES, candidate_axes
from scattermesh.curriculum_v2 import generate_lineages
from scattermesh.dataset_v2 import build_new_dataset
from scattermesh.scoring_v2 import penalty_sensitivity, rank_candidates, soft_dt_score


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


def test_deadline_creates_resumable_incomplete_records(tmp_path):
    scene = generate_lineages(128)[0]
    manifest = {"scenes": [scene], "incidence_angles": [0.0], "budgets": [32]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    counts = mark_unfinished(path, tmp_path / "campaign")
    assert counts == {"references": 1, "cases": len(CANDIDATE_NAMES)}
    assert mark_unfinished(path, tmp_path / "campaign") == {"references": 0, "cases": 0}
