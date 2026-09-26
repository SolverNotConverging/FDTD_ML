"""Collection geometry, scene inputs, and two-object numerical regressions."""

import numpy as np
import pytest

from scattermesh import Grid, PlaneWave, simulate, simulate_cuda
from scattermesh.candidates_v2 import candidate_axes
from scattermesh.curriculum_v2 import (
    c4_gap_sampling_diagnostics,
    conditioning_v2,
    generate_c4_gap_sweep,
    generate_c8_c9_targets,
    generate_compact_poc_scenes,
    generate_development_scenes,
    generate_multiobject_lineages,
    objects_from_scene,
    rasterize_v2,
    scene_material_kind,
    scene_metrics,
)
from scattermesh.gap_sweep_v2 import _gap_sample_points
from scattermesh.observables import field_point_stencil, interpolate_field_points


def _pair_scene():
    return {
        "schema_version": 3,
        "lineage_id": "pair_axis_separated",
        "family": "multi_object",
        "stage": "C3",
        "object_families": ["circle", "rectangle"],
        "objects": [
            {
                "object_id": "left",
                "shape": "circle",
                "center_m": [0.36, 0.40],
                "radius_m": 0.05,
                "material": {"kind": "dielectric", "epsilon_r": 2.0},
            },
            {
                "object_id": "right",
                "shape": "rectangle",
                "bounds_m": [0.78, 0.88, 0.74, 0.84],
                "material": {"kind": "dielectric", "epsilon_r": 4.0},
            },
        ],
        "split": "train",
    }


def test_multiobject_generator_is_grouped_bounded_and_deterministic():
    scenes = generate_multiobject_lineages(64, seed=21)
    assert scenes == generate_multiobject_lineages(64, seed=21)
    assert len({scene["lineage_id"] for scene in scenes}) == 64
    assert {scene["stage"] for scene in scenes} == {"C3", "C5"}
    assert {scene["split"] for scene in scenes} == {"train", "validation", "test"}
    for scene in scenes:
        metrics = scene_metrics(scene)
        assert 2 <= metrics["object_count"] <= 5
        assert metrics["occupied_area_fraction"] > 0
        assert metrics["projected_x_support_fraction"] < metrics["projected_x_extent_fraction"] or (
            metrics["projected_y_support_fraction"] < metrics["projected_y_extent_fraction"]
        )
        assert len(objects_from_scene(scene)) == metrics["object_count"]
        assert scene_material_kind(scene) == "dielectric"


def test_extended_collection_generator_covers_six_to_ten_mixed_dielectrics():
    scenes = generate_multiobject_lineages(
        64,
        seed=29,
        object_counts=range(2, 11),
        mixed_materials=True,
    )
    assert {scene["stage"] for scene in scenes} == {"C3", "C5"}
    assert max(len(scene["objects"]) for scene in scenes) == 10
    extended = [scene for scene in scenes if 6 <= len(scene["objects"]) <= 10]
    assert extended
    for scene in extended:
        metrics = scene_metrics(scene)
        assert metrics["object_count"] == len(scene["objects"])
        assert metrics["minimum_axis_aligned_bbox_gap_m"] > 0
        assert {item["material"]["epsilon_r"] for item in scene["objects"]} == {2.0, 4.0}
        assert scene["material_assignment"] == "per_object_epsilon_2_or_4"
        assert scene_material_kind(scene) == "dielectric"
    ten_object_scene = next(scene for scene in extended if len(scene["objects"]) == 10)
    assert rasterize_v2(ten_object_scene, 64).shape == (9, 64, 64)
    x, y, _ = candidate_axes(ten_object_scene, 32, "interface")
    assert len(x) == len(y) == 33


def test_collection_raster_conditioning_metrics_and_candidates_use_all_objects():
    scene = _pair_scene()
    metrics = scene_metrics(scene)
    assert metrics["object_count"] == 2
    assert metrics["occupied_area_fraction"] == pytest.approx(
        (np.pi * 0.05**2 + 0.1 * 0.1) / 1.2**2
    )
    assert metrics["projected_x_support_fraction"] == pytest.approx((0.1 + 0.1) / 1.2)
    assert metrics["projected_x_extent_fraction"] > metrics["projected_x_support_fraction"]
    assert scene_material_kind(scene) == "dielectric"

    raster = rasterize_v2(scene, 96)
    assert raster.shape == (9, 96, 96)
    assert raster[0].max() > 0
    assert raster[6].max() > 0
    assert np.isfinite(raster).all()
    condition = conditioning_v2(scene, 48, 64, 0.2)
    assert condition.shape == (15,)
    assert condition[10] == pytest.approx(0.2)
    assert condition[12] == pytest.approx(metrics["occupied_area_fraction"])
    assert np.isfinite(condition).all()

    x, y, repairs = candidate_axes(scene, 48, "interface")
    assert len(x) == len(y) == 49
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    assert all(0 <= repair <= 1 for repair in repairs)


def test_interface_policy_uses_internal_concave_boundary_vertices():
    scene = {
        "schema_version": 2,
        "lineage_id": "concave_interface_anchors",
        "family": "concave_polygon",
        "stage": "C2",
        "shape": "concave_polygon",
        "vertices_m": [
            [0.3, 0.3],
            [0.8, 0.3],
            [0.8, 0.8],
            [0.6, 0.8],
            [0.6, 0.5],
            [0.5, 0.5],
            [0.5, 0.8],
            [0.3, 0.8],
        ],
        "material": {"kind": "dielectric", "epsilon_r": 2.0},
    }
    anchors = scene_metrics(scene)["boundary_coordinates"][0]
    assert 0.5 in anchors and 0.6 in anchors
    x, _, _ = candidate_axes(scene, 48, "interface")
    assert len(x) == 49
    assert np.std(np.diff(x)) > 0


def test_compact_poc_splits_cover_bridge_and_heldout_targets_without_lineage_leakage():
    scenes = generate_compact_poc_scenes(seed=31)
    assert scenes == generate_compact_poc_scenes(seed=31)
    assert len(scenes) == 104
    assert {
        split: sum(row["split"] == split for row in scenes)
        for split in ("train", "validation", "test")
    } == {
        "train": 64,
        "validation": 16,
        "test": 24,
    }
    assert {row["stage"] for row in scenes if row["split"] == "train"} >= {
        "C0",
        "C1",
        "C2",
        "C3",
        "C5",
    }
    assert all(row["stage"] in {"C8", "C9"} for row in scenes if row["split"] == "test")
    assert all("template_id" in row for row in scenes if row["stage"] == "C8")
    assert all(2 <= len(row["objects"]) <= 6 for row in scenes if row["stage"] == "C9")
    assert len({row["lineage_id"] for row in scenes}) == len(scenes)


def test_development_templates_are_separate_from_frozen_c8_target_templates():
    development = generate_development_scenes()
    targets = generate_c8_c9_targets()
    development_templates = {row["template_id"] for row in development if "template_id" in row}
    target_templates = {row["template_id"] for row in targets if row["stage"] == "C8"}
    assert len(development) == 8
    assert len(development_templates) == 3
    assert development_templates.isdisjoint(target_templates)
    assert {row["stage"] for row in development} >= {"C0", "C2", "C3", "C5", "C8"}


def test_c4_gap_sweep_preserves_positive_continuous_gaps_and_reports_sampling():
    gaps = (0.002, 0.005, 0.01, 0.02)
    scenes = generate_c4_gap_sweep(gaps)

    assert [scene["gap_m"] for scene in scenes] == list(gaps)
    for scene in scenes:
        assert scene["stage"] == "C4"
        assert scene["geometry_qualification"] == "continuous_positive_gap_only"
        assert len(scene["objects"]) == 2
        assert np.isclose(scene_metrics(scene)["minimum_axis_aligned_bbox_gap_m"], scene["gap_m"])
        assert all(obj.material.epsilon_r == 4.0 for obj in objects_from_scene(scene))

    smallest = c4_gap_sampling_diagnostics(scenes[0], 48)
    largest = c4_gap_sampling_diagnostics(scenes[-1], 48)
    assert smallest["subcell_gap"] and smallest["subpixel_gap"]
    assert smallest["qualification_status"] == "requires_convergence_study"
    assert largest["gap_in_uniform_cells"] > smallest["gap_in_uniform_cells"]


def test_subcell_c4_dielectric_gap_runs_without_geometry_repair():
    scene = generate_c4_gap_sweep((0.002,))[0]
    objects = objects_from_scene(scene)
    grid = Grid(np.linspace(0, 1.2, 49), np.linspace(0, 1.2, 49))
    source = PlaneWave(1e9, 1e-9, 9e-9, origin=(0.6, 0.6))

    result = simulate(
        grid,
        objects,
        source,
        frequencies=[1e9],
        duration=12e-9,
        pml_thickness=0.12,
        monitor_bounds=(0.225, 0.975, 0.225, 0.975),
        samples=12,
    )

    assert np.isfinite(result.fields["Ez"]).all()
    assert np.isfinite(
        result.monitor.normalized_far_field(np.linspace(0, 2 * np.pi, 36, endpoint=False))
    ).all()


def test_engineering_targets_make_legal_exact_budget_axes():
    scenes = generate_c8_c9_targets()
    for scene in scenes:
        scene_metrics(scene)
        x, y, repairs = candidate_axes(scene, 32, "interface")
        assert len(x) == len(y) == 33
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
        for axis in (x, y):
            widths = np.diff(axis)
            assert (
                max(np.max(widths[1:] / widths[:-1]), np.max(widths[:-1] / widths[1:])) <= 3 + 1e-10
            )
        assert all(0 <= repair <= 1 for repair in repairs)


def test_aircraft_like_c8_outline_runs_through_existing_tmz_solver():
    scene = generate_c8_c9_targets(silhouette_count=3, distributed_count=3)[0]
    objects = objects_from_scene(scene)
    grid = Grid(np.linspace(0, 1.2, 49), np.linspace(0, 1.2, 49))
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.2, origin=(0.6, 0.6))
    settings = {
        "frequencies": [1e9],
        "duration": 12e-9,
        "pml_thickness": 0.15,
        "monitor_bounds": (0.225, 0.975, 0.225, 0.975),
        "samples": 4,
    }
    result = simulate(grid, objects, source, **settings)
    assert np.isfinite(result.fields["Ez"]).all()
    assert np.isfinite(
        result.monitor.normalized_far_field(np.linspace(0, 2 * np.pi, 36, endpoint=False))
    ).all()


def test_collection_rejects_overlapping_bounds_and_duplicate_ids():
    scene = _pair_scene()
    scene["objects"][1]["bounds_m"] = [0.39, 0.49, 0.38, 0.48]
    with pytest.raises(ValueError, match="disjoint bounds"):
        scene_metrics(scene)
    scene = _pair_scene()
    scene["objects"][1]["object_id"] = "left"
    with pytest.raises(ValueError, match="unique object_id"):
        objects_from_scene(scene)


def test_two_object_scene_uses_existing_cpu_cuda_solver_consistently():
    scene = _pair_scene()
    objects = objects_from_scene(scene)
    grid = Grid(np.linspace(0, 1.2, 33), np.linspace(0, 1.2, 33))
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.4, origin=(0.6, 0.6))
    settings = {
        "frequencies": [1e9],
        "duration": 12e-9,
        "pml_thickness": 0.15,
        "monitor_bounds": (0.225, 0.975, 0.225, 0.975),
        "samples": 4,
        "field_sample_points": np.array([[0.45, 0.6], [0.6, 0.6], [0.74, 0.6]]),
    }
    reference = simulate(grid, objects, source, **settings)
    candidate = simulate_cuda(grid, objects, source, device="cpu", dtype="float64", **settings)
    np.testing.assert_allclose(
        candidate.fields["Ez"], reference.fields["Ez"], rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(
        candidate.monitor.normalized_far_field(np.linspace(0, 2 * np.pi, 36, endpoint=False)),
        reference.monitor.normalized_far_field(np.linspace(0, 2 * np.pi, 36, endpoint=False)),
        rtol=2e-12,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        candidate.fields["Ez_point_scattered_dft"],
        reference.fields["Ez_point_scattered_dft"],
        rtol=2e-12,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        candidate.fields["Ez_point_incident_dft"],
        reference.fields["Ez_point_incident_dft"],
        rtol=2e-12,
        atol=2e-14,
    )


def test_field_point_stencil_bilinearly_interpolates_on_a_nonuniform_grid():
    x = np.array([0.0, 0.1, 0.35, 0.72, 1.2])
    y = np.array([0.0, 0.2, 0.43, 0.81, 1.2])
    grid = Grid(x, y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = 1.3 * xx - 0.7 * yy + 2.0
    points = np.array([[0.23, 0.61], [0.99, 0.31]])
    actual = interpolate_field_points(field, field_point_stencil(grid, points))
    np.testing.assert_allclose(actual, 1.3 * points[:, 0] - 0.7 * points[:, 1] + 2.0)
    with pytest.raises(ValueError, match="in the grid"):
        field_point_stencil(grid, [[1.21, 0.4]])


@pytest.mark.parametrize("gap", [0.002, 0.005, 0.01, 0.02])
def test_c4_near_field_samples_lie_inside_the_continuous_gap(gap):
    scene = generate_c4_gap_sweep([gap])[0]
    points = _gap_sample_points(scene)
    left_center, right_center = [np.asarray(obj["center_m"]) for obj in scene["objects"]]
    radius = scene["objects"][0]["radius_m"]
    x_left, x_right = left_center[0] + radius, right_center[0] - radius
    assert points.shape == (5, 2)
    assert np.all((points[:, 0] > x_left) & (points[:, 0] < x_right))
    np.testing.assert_allclose(points[:, 1], left_center[1])


def test_disjoint_collection_material_average_is_object_order_invariant():
    scene = _pair_scene()
    forward = objects_from_scene(scene)
    reverse = tuple(reversed(forward))
    grid = Grid(np.linspace(0, 1.2, 33), np.linspace(0, 1.2, 33))
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.4, origin=(0.6, 0.6))
    settings = {
        "frequencies": [1e9],
        "duration": 12e-9,
        "pml_thickness": 0.15,
        "monitor_bounds": (0.225, 0.975, 0.225, 0.975),
        "samples": 4,
    }
    first = simulate(grid, forward, source, **settings)
    second = simulate(grid, reverse, source, **settings)
    np.testing.assert_allclose(first.fields["Ez"], second.fields["Ez"], rtol=0, atol=0)
