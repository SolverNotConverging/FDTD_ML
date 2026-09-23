"""C0--C2 split, physics-score, and mesh-budget contracts."""

from collections import Counter

import numpy as np
import pytest

from scattermesh.curriculum_v2 import (
    conditioning_v2,
    generate_lineages,
    object_from_scene,
    rasterize_v2,
    scene_metrics,
)
from scattermesh.scoring_v2 import penalty_sensitivity, rank_candidates, threshold_cell_savings


def test_scene_families_are_split_by_independent_lineage():
    scenes = generate_lineages(256)
    assert len(scenes) == len({row["lineage_id"] for row in scenes}) == 256
    assert Counter(row["split"] for row in scenes) == {"train": 192, "validation": 32, "test": 32}
    assert Counter(row["material"]["kind"] for row in scenes) == {"dielectric": 192, "pec": 64}
    assert {row["stage"] for row in scenes} == {"C0", "C1", "C2"}
    for row in scenes:
        metrics = scene_metrics(row)
        assert 0.005 <= metrics["occupied_area_fraction"] <= 0.08
        assert (
            max(metrics["projected_x_support_fraction"], metrics["projected_y_support_fraction"])
            <= 0.35
        )
        assert object_from_scene(row).contains(*row["center_m"])
    assert generate_lineages(256) == scenes


def test_raster_encodes_continuous_material_and_spatial_position():
    scene = next(
        row
        for row in generate_lineages(128)
        if row["family"] == "triangle" and row["material"]["kind"] == "dielectric"
    )
    raster = rasterize_v2(scene, 64)
    assert raster.shape == (9, 64, 64)
    assert raster.dtype == np.float32
    assert raster[0].max() > 0 and raster[3].max() == 0
    assert np.all(raster[7, :, 1:] > raster[7, :, :-1])
    assert np.all(raster[8, 1:, :] > raster[8, :-1, :])
    features = conditioning_v2(scene, 32, 48, 0.7)
    assert features.shape == (15,)
    assert np.isfinite(features).all()


def test_weak_dt_penalty_keeps_large_accuracy_gain():
    rows = [
        {"name": "uniform", "accepted": True, "joint_scattering_loss": 0.1, "dt": 1e-11},
        {"name": "focused", "accepted": True, "joint_scattering_loss": 0.04, "dt": 1e-12},
        {"name": "failed", "accepted": False, "joint_scattering_loss": 0.001, "dt": 1e-12},
    ]
    assert rank_candidates(rows)[0]["name"] == "focused"
    assert rank_candidates(rows)[0]["teacher_score"] == pytest.approx(0.04 * 10**0.05)
    assert penalty_sensitivity(rows)["0.1"]["best_candidate"] == "focused"
    assert (
        threshold_cell_savings(
            [
                {
                    "accepted": True,
                    "cells_x": 96,
                    "cells_y": 96,
                    "complex_relative_l2": 0.018,
                    "width_relative_l2": 0.019,
                }
            ],
            [
                {
                    "accepted": True,
                    "cells_x": 64,
                    "cells_y": 64,
                    "complex_relative_l2": 0.015,
                    "width_relative_l2": 0.016,
                }
            ],
        )["0.02"]["cell_saving_ratio"]
        == 2.25
    )
