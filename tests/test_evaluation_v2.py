"""Frozen target evaluation can qualify C8 scenes without teacher labels."""

import json

import numpy as np

from scattermesh import evaluation_v2
from scattermesh.curriculum_v2 import generate_c8_c9_targets


def test_frozen_evaluation_creates_target_reference_and_uniform_baseline(tmp_path, monkeypatch):
    scene = generate_c8_c9_targets(silhouette_count=3, distributed_count=3)[0]
    manifest = {
        "scenes": [scene],
        "incidence_angles": [0.0],
        "budgets": [32],
        "frequencies_hz": [0.8e9, 1.0e9, 1.2e9],
        "evaluation_conditions": {"test": {scene["lineage_id"]: [{"angle_index": 0, "cells": 32}]}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint_path = tmp_path / "selected.pt"
    checkpoint_path.write_bytes(b"frozen")
    campaign = tmp_path / "campaign"
    calls = []

    monkeypatch.setattr(
        evaluation_v2, "load_checkpoint", lambda path, device: (object(), {"epoch": 12})
    )

    def qualify(target, angle_index, angle, output, **kwargs):
        path = output / "qualifications" / f"{target['lineage_id']}_a{angle_index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        field = np.ones((3, 8), dtype=np.complex128)
        np.savez(path.with_suffix(".npz"), complex_far_field=field)
        report = {
            "accepted": True,
            "selected_cells": 256,
            "reference_uncertainty_relative": 0.002,
        }
        path.write_text(json.dumps(report))
        calls.append(("qualify", target["stage"]))
        return report

    def evaluate_case(target, cells, angle, policy, output, **kwargs):
        calls.append((policy, target["stage"]))
        return {
            "accepted": True,
            "joint_scattering_loss": 0.1 if policy == "uniform" else 0.08,
        }, False

    monkeypatch.setattr(evaluation_v2, "qualify_reference", qualify)
    monkeypatch.setattr(evaluation_v2, "evaluate_case", evaluate_case)
    monkeypatch.setattr(
        evaluation_v2,
        "predict_axes",
        lambda *args: ((np.linspace(0, 1.2, 33), np.linspace(0, 1.2, 33)), (0.0, 0.0)),
    )

    result = evaluation_v2.evaluate_checkpoint(
        manifest_path,
        campaign,
        checkpoint_path,
        tmp_path / "evaluation",
        split="test",
        device="cpu",
    )
    assert result["terminal"]
    assert result["requested_conditions"] == result["valid_conditions"] == 1
    assert calls == [
        ("qualify", "C8"),
        ("uniform", "C8"),
        ("interface", "C8"),
        ("learned", "C8"),
    ]
