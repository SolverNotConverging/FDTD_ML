import numpy as np
import pytest

from scattermesh import (
    Grid,
    circle_cluster_objects,
    sparse_circle_candidate_axes,
    sparse_cluster_metrics,
    sparse_scene_objects,
)


def scene(gap=0.02):
    radius = 0.04
    return {
        "objects": [
            {
                "shape": "circle",
                "center_m": [0.60 - radius - gap / 2, 0.60],
                "radius_m": radius,
                "material": {"kind": "dielectric", "epsilon_r": 4, "sigma_e_s_per_m": 0.02},
            },
            {
                "shape": "circle",
                "center_m": [0.60 + radius + gap / 2, 0.60],
                "radius_m": radius,
                "material": {"kind": "dielectric", "epsilon_r": 8, "sigma_e_s_per_m": 0.04},
            },
        ]
    }


def test_sparse_cluster_metrics_measure_gap_area_and_projected_support():
    metrics = sparse_cluster_metrics(scene())
    assert metrics["object_count"] == 2
    assert metrics["pec_fraction"] == 0
    assert metrics["minimum_gap_m"] == pytest.approx(0.02)
    assert metrics["occupied_area_fraction"] == pytest.approx(2 * np.pi * 0.04**2 / 1.2**2)
    assert metrics["projected_x_support_fraction"] == pytest.approx(0.16 / 1.2)
    assert metrics["projected_y_support_fraction"] == pytest.approx(0.08 / 1.2)


@pytest.mark.parametrize(
    "policy",
    [
        {"kind": "uniform"},
        {"kind": "interface", "interface_width_factor": 0.5, "interface_weight": 1.5},
        {"kind": "cluster", "cluster_width_factor": 0.8, "cluster_weight": 1.0},
        {"kind": "gap", "gap_width_factor": 1.0, "gap_min_width_m": 0.01, "gap_weight": 2.0},
        {
            "kind": "hybrid",
            "interface_width_factor": 0.5,
            "interface_weight": 1.0,
            "cluster_width_factor": 0.8,
            "cluster_weight": 0.5,
            "gap_width_factor": 1.0,
            "gap_min_width_m": 0.01,
            "gap_weight": 1.5,
        },
    ],
)
def test_sparse_candidate_axes_preserve_exact_budget_and_grading(policy):
    x, y = sparse_circle_candidate_axes(1.2, 32, scene(), policy)
    grid = Grid(x, y, max_ratio=3.0 if policy["kind"] != "uniform" else 1.0)
    assert grid.shape == (33, 33)


def test_sparse_cluster_rejects_touching_objects():
    with pytest.raises(ValueError, match="positive gap"):
        circle_cluster_objects(scene(gap=0.0))


def test_mixed_circle_rectangle_pec_metrics_and_gap_mesh():
    mixed = {
        "objects": [
            {
                "shape": "circle",
                "center_m": [0.48, 0.60],
                "radius_m": 0.05,
                "material": {"kind": "pec"},
            },
            {
                "shape": "rectangle",
                "bounds_m": [0.55, 0.67, 0.54, 0.66],
                "material": {
                    "kind": "dielectric",
                    "epsilon_r": 6,
                    "sigma_e_s_per_m": 0.04,
                },
            },
        ]
    }
    objects = sparse_scene_objects(mixed)
    assert len(objects) == 2
    metrics = sparse_cluster_metrics(mixed)
    assert metrics["minimum_gap_m"] == pytest.approx(0.02)
    assert metrics["pec_fraction"] == 0.5
    assert metrics["shape_topology"] == "circle+rectangle"
    assert metrics["material_topology"] == "dielectric+pec"
    policy = {
        "kind": "hybrid",
        "interface_width_factor": 0.5,
        "interface_weight": 1.0,
        "cluster_width_factor": 0.8,
        "cluster_weight": 0.5,
        "gap_width_factor": 1.0,
        "gap_min_width_m": 0.01,
        "gap_weight": 1.5,
    }
    x, y = sparse_circle_candidate_axes(1.2, 48, mixed, policy)
    assert Grid(x, y, max_ratio=3).shape == (49, 49)


def test_rectangle_pair_rejects_overlap():
    overlapping = {
        "objects": [
            {"shape": "rectangle", "bounds_m": [0.4, 0.6, 0.4, 0.6], "material": {"kind": "pec"}},
            {"shape": "rectangle", "bounds_m": [0.5, 0.7, 0.5, 0.7], "material": {"kind": "pec"}},
        ]
    }
    with pytest.raises(ValueError, match="positive gap"):
        sparse_scene_objects(overlapping)
