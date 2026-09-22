import json

import numpy as np
import pytest

from scattermesh.distillation import (
    axis_probability,
    build_distillation_dataset,
    conditioning_features,
    probability_axis,
    rasterize_circle,
)


def circle_example(cells=32):
    return {
        "shape": "circle",
        "center_m": [0.57, 0.63],
        "radius_m": 0.1,
        "feature_size_m": 0.2,
        "epsilon_r": 20.0,
        "sigma_e_s_per_m": 0.12,
        "incidence_angle_rad": 0.7,
        "frequencies_hz": [0.8e9, 1.0e9, 1.2e9],
        "cells_x": cells,
        "cells_y": cells,
    }


def test_axis_probability_roundtrip_and_grading_repair():
    axis = np.linspace(0, 1.2, 49)
    probability = axis_probability(axis, 128)
    reconstructed, repair = probability_axis(probability, 48)
    assert repair == 0
    assert reconstructed == pytest.approx(axis, abs=1e-12)

    spike = np.full(128, 1e-12)
    spike[63:65] = 1
    repaired, repair = probability_axis(spike, 32, max_ratio=3)
    spacing = np.diff(repaired)
    grading = max(np.max(spacing[1:] / spacing[:-1]), np.max(spacing[:-1] / spacing[1:]))
    assert len(repaired) == 33
    assert repaired[0] == 0 and repaired[-1] == 1.2
    assert repair > 0
    assert grading <= 3 * (1 + 1e-12)


def test_circle_raster_and_conditioning_include_feature_and_budget():
    example = circle_example()
    raster = rasterize_circle(example, 64)
    features = conditioning_features(example)
    assert raster.shape == (5, 64, 64)
    assert np.isfinite(raster).all()
    assert raster[0].max() == 1
    assert raster[1].max() > 0
    assert raster[2].max() > 0
    assert features.shape == (10,)
    assert features[2] == pytest.approx(0.25)
    assert features[6] == pytest.approx(1 / 6)


def test_model_returns_normalized_axis_probabilities():
    torch = pytest.importorskip("torch")
    from scattermesh.model import AxisDensityUNet, set_valued_profile_loss

    model = AxisDensityUNet(base_channels=8)
    prediction = model(torch.randn(2, 5, 32, 32), torch.randn(2, 10))
    assert prediction.shape == (2, 2, 32)
    assert torch.all(prediction > 0)
    assert torch.allclose(prediction.sum(dim=-1), torch.ones(2, 2))
    targets = torch.softmax(torch.randn(2, 3, 2, 32), dim=-1)
    scores = torch.tensor([[1.0, 1.2, 2.0], [1.3, 1.0, 1.5]])
    loss, components = set_valued_profile_loss(
        prediction, targets, scores, torch.tensor([0, 1])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(components) == {"selected_cdf_l1", "weighted_set_loss"}


def test_dataset_builder_requires_and_preserves_exact_accepted_candidates(tmp_path):
    geometry = {
        "geometry_id": "g1",
        "lineage_id": "lineage1",
        "split": "train",
        "family": "simple",
        **circle_example(),
    }
    condition = {
        "task_id": "c1",
        "geometry_id": "g1",
        "lineage_id": "lineage1",
        "split": "train",
        "family": "simple",
        "illumination_id": "i1",
        "incidence_angle_rad": 0.7,
        "cells_x": 32,
        "cells_y": 32,
    }
    manifest = {
        "dataset_id": "dataset1",
        "geometries": [geometry],
        "conditions": [condition],
    }
    campaign = {
        "dataset_id": "dataset1",
        "campaign_id": "campaign1",
        "condition_ids": ["c1"],
        "candidate_names": ["uniform", "region"],
        "ranking": {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1},
    }
    manifest_path, campaign_path = tmp_path / "manifest.json", tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest))
    campaign_path.write_text(json.dumps(campaign))
    run = tmp_path / "run"
    (run / "cases").mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps(
            {
                "campaign_id": "campaign1",
                "decision": "accepted",
                "case_count": 2,
                "status_counts": {"accepted": 2},
                "training_readiness": {
                    "decision": "ready_for_m5_pilot",
                    "checks": {
                        "campaign_complete": True,
                        "train_has_multiple_winning_lineages": True,
                        "train_has_multiple_winning_candidates": True,
                        "train_has_multiple_winning_budgets": True,
                        "validation_has_headroom": True,
                        "test_has_headroom": True,
                    },
                },
            }
        )
    )
    (run / "labels.json").write_text(
        json.dumps({"groups": {"i1": {"budgets": {"32": {"best_case": "c1_region"}}}}})
    )
    axes = {
        "uniform": np.linspace(0, 1.2, 33),
        "region": np.r_[np.linspace(0, 0.5, 12), np.linspace(0.52, 1.2, 21)],
    }
    for candidate, loss, nt in (("uniform", 0.2, 100), ("region", 0.1, 120)):
        directory = run / "cases" / f"c1_{candidate}"
        directory.mkdir()
        (directory / "record.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "config": {"cells": 32},
                    "joint_scattering_loss": loss,
                    "Nt": nt,
                }
            )
        )
        np.savez(directory / "spectra.npz", x=axes[candidate], y=axes[candidate])

    payload = build_distillation_dataset(
        manifest_path, campaign_path, run, tmp_path / "dataset", profile_bins=64
    )
    assert payload["example_count"] == 1
    assert payload["split_counts"] == {"train": 1}
    assert payload["examples"][0]["best_candidate"] == "region"
    with np.load(tmp_path / "dataset" / "targets.npz") as arrays:
        assert arrays["profiles"].shape == (1, 2, 2, 64)
        assert arrays["scores"][0, 1] < arrays["scores"][0, 0]

    broken = json.loads((run / "report.json").read_text())
    broken["decision"] = "incomplete_uniform_baselines"
    (run / "report.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="accepted report"):
        build_distillation_dataset(manifest_path, campaign_path, run, tmp_path / "broken")

    broken["decision"] = "accepted"
    broken["training_readiness"]["checks"]["validation_has_headroom"] = False
    broken["training_readiness"]["decision"] = "not_ready_for_m5"
    (run / "report.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="label-diversity gate"):
        build_distillation_dataset(manifest_path, campaign_path, run, tmp_path / "not_ready")
