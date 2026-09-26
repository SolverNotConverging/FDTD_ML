"""Continuous polygon-ring geometry checks for the C6 topology stage."""

import json

import numpy as np
import pytest

from scattermesh import (
    Grid,
    PlaneWave,
    PolygonWithHoles,
    c6_qualification_v2,
    simulate,
    simulate_cuda,
)
from scattermesh.candidates_v2 import candidate_axes
from scattermesh.conformal import CutCellPEC
from scattermesh.curriculum_v2 import (
    _scaled_definition,
    object_from_definition,
    rasterize_v2,
    scene_metrics,
)
from scattermesh.topology_v2 import open_cavity_smoke_scene, square_ring_smoke_scene


def test_polygon_hole_membership_line_intervals_and_feature_size():
    definition = square_ring_smoke_scene()["objects"][0]
    ring = object_from_definition(definition)
    assert isinstance(ring, PolygonWithHoles)
    assert ring.contains(0.5, 0.6)
    assert not ring.contains(0.6, 0.6)
    assert not ring.contains(0.4, 0.6)
    assert ring.line_intervals(0.6, axis=1) == pytest.approx([(0.45, 0.54), (0.66, 0.75)])
    assert ring.minimum_feature_size == pytest.approx(0.09)

    metrics = scene_metrics(square_ring_smoke_scene())
    assert metrics["occupied_area_fraction"] * 1.2**2 == pytest.approx(0.0756)
    assert metrics["feature_size_m"] == pytest.approx(0.09)


def test_polygon_hole_is_preserved_by_rasterization():
    raster = rasterize_v2(square_ring_smoke_scene(), resolution=120)
    assert raster.shape == (9, 120, 120)
    assert raster[0, 60, 60] == 0
    assert raster[0, 50, 60] > 0.99
    assert raster[1, 60, 60] == 0


def test_polygon_hole_requires_schema_four():
    scene = square_ring_smoke_scene()
    scene["schema_version"] = 3
    with pytest.raises(ValueError, match="requires scene schema version 4"):
        scene_metrics(scene)


def test_transforms_preserve_ring_offsets_and_scale_the_minimum_feature():
    definition = square_ring_smoke_scene()["objects"][0]
    moved = _scaled_definition(definition, (0.7, 0.7), 0.5, definition["material"])
    ring = object_from_definition(moved)
    assert ring.bounds == pytest.approx((0.625, 0.775, 0.625, 0.775))
    assert ring.minimum_feature_size == pytest.approx(0.045)
    assert moved["feature_size_m"] == pytest.approx(0.1)


def test_c6_campaign_covers_three_wall_scales_and_open_cavity():
    scenes = c6_qualification_v2.generate_c6_topology_scenes()
    assert len(scenes) == 4
    rings = [scene for scene in scenes if scene["family"] == "topology_ring"]
    assert {scene["ring_wall_thickness_m"] for scene in rings} == {0.09, 0.03, 0.015}
    cavity = next(scene for scene in scenes if scene["family"] == "topology_open_cavity")
    assert scene_metrics(cavity)["feature_size_m"] == pytest.approx(0.12)
    assert all(scene["objects"][0]["material"]["sigma_e_s_per_m"] == 0.002 for scene in scenes)


def test_c6_campaign_records_qualified_and_failed_candidates(tmp_path, monkeypatch):
    def fake_reference(scene, root, **kwargs):
        reference = np.ones((3, 180), dtype=np.complex128)
        return {
            "accepted": True,
            "status": "accepted",
            "selected_cells": 64,
            "reference_uncertainty_relative": 0.001,
            "independent_probes": {},
        }, reference

    def fake_evaluate(scene, cells, angle, policy, output, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        accepted = policy == "uniform" or cells > 32
        field = np.ones((3, 180), dtype=np.complex128) * (1.0 if policy == "uniform" else 0.99)
        np.savez_compressed(output / "spectra.npz", complex_far_field=field)
        record = {
            "status": "accepted" if accepted else "unsettled",
            "accepted": accepted,
            "joint_scattering_loss": 0.0 if policy == "uniform" else 0.01,
            "complex_relative_l2": 0.0 if policy == "uniform" else 0.01,
            "width_relative_l2": 0.0 if policy == "uniform" else 0.01,
            "dt": 1e-11,
            "Nt": 100,
            "wall_seconds": 1.0,
            "uniform_repair_fraction": [0.0, 0.0],
        }
        (output / "record.json").write_text(json.dumps(record))
        return record, False

    monkeypatch.setattr(c6_qualification_v2, "_qualify_reference", fake_reference)
    monkeypatch.setattr(c6_qualification_v2, "evaluate_case", fake_evaluate)
    report = c6_qualification_v2.run_c6_topology_qualification(
        tmp_path,
        reference_levels=(64, 96),
        budgets=(32,),
        policies=("uniform", "hybrid"),
    )
    assert report["qualified_scene_count"] == 4
    assert report["candidate_case_count"] == 8
    assert report["candidate_status_counts"] == {"accepted": 4, "unsettled": 4}
    assert (tmp_path / "c6_topology_qualification_report.json").is_file()


def test_topology_interface_policy_refines_around_inner_rings():
    scene = square_ring_smoke_scene()
    uniform_x, _, _ = candidate_axes(scene, 64, "uniform")
    hybrid_x, hybrid_y, _ = candidate_axes(scene, 64, "hybrid")
    assert len(hybrid_x) == len(hybrid_y) == 65
    for axis in (hybrid_x, hybrid_y):
        assert min(abs(axis - 0.54)) < min(abs(uniform_x - 0.54))
        assert min(abs(axis - 0.66)) < min(abs(uniform_x - 0.66))
        assert np.all(np.diff(axis) > 0)
    grid = Grid(hybrid_x, hybrid_y, max_ratio=3)
    assert grid.diagnostics()["x_grading"] <= 3 + 1e-12
    assert grid.diagnostics()["y_grading"] <= 3 + 1e-12


@pytest.mark.parametrize("wall", [0.09, 0.03, 0.015])
def test_ring_raster_and_mesh_input_preserve_thin_wall_features(wall):
    scene = square_ring_smoke_scene(wall_thickness_m=wall)
    metrics = scene_metrics(scene)
    assert metrics["feature_size_m"] == pytest.approx(wall)
    raster = rasterize_v2(scene, resolution=512)
    assert raster[0, 256, 194] > 0
    assert raster[0, 256, 256] == 0

    uniform_x, _, _ = candidate_axes(scene, 96, "uniform")
    hybrid_x, _, _ = candidate_axes(scene, 96, "hybrid")
    inner_edge = 0.45 + wall
    assert min(abs(hybrid_x - inner_edge)) < min(abs(uniform_x - inner_edge))


def test_open_cavity_remains_void_and_has_two_exact_boundary_intervals():
    scene = open_cavity_smoke_scene()
    cavity = object_from_definition(scene["objects"][0])
    assert cavity.contains(0.6, 0.45)
    assert not cavity.contains(0.6, 0.65)
    assert cavity.line_intervals(0.65, axis=0) == pytest.approx([(0.4, 0.52), (0.68, 0.8)])
    raster = rasterize_v2(scene, resolution=120)
    assert raster[0, 65, 60] == 0
    assert raster[0, 45, 60] > 0.99


@pytest.mark.parametrize(
    "hole, message",
    [
        ([[0.70, 0.54], [0.78, 0.54], [0.78, 0.66], [0.70, 0.66]], "inside"),
        ([[0.45, 0.54], [0.54, 0.54], [0.54, 0.66], [0.45, 0.66]], "touch or intersect"),
    ],
)
def test_invalid_or_touching_hole_rings_are_rejected(hole, message):
    definition = square_ring_smoke_scene()["objects"][0]
    definition["holes_m"] = [hole]
    with pytest.raises(ValueError, match=message):
        object_from_definition(definition)


def test_dielectric_ring_runs_through_the_tmz_solver():
    ring = object_from_definition(square_ring_smoke_scene()["objects"][0])
    axis = np.linspace(0.0, 1.2, 65)
    result = simulate(
        Grid(axis, axis),
        [ring],
        PlaneWave(1e9, 1e-9, 9e-9),
        frequencies=[1e9],
        duration=560e-9,
        pml_thickness=0.12,
    )
    assert np.isfinite(result.monitor.far_amplitude(np.linspace(0, 2 * np.pi, 32))[0]).all()
    assert result.diagnostics["tail_peak_over_global_peak"] < 1e-5


@pytest.mark.parametrize("scene_factory", [square_ring_smoke_scene, open_cavity_smoke_scene])
def test_topology_cuda_matches_cpu_far_field(scene_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not visible")
    obj = object_from_definition(scene_factory()["objects"][0])
    axis = np.linspace(0.0, 1.2, 33)
    grid = Grid(axis, axis)
    source = PlaneWave(1e9, 1e-9, 9e-9)
    kwargs = {"frequencies": [0.8e9, 1e9, 1.2e9], "duration": 12e-9, "pml_thickness": 0.12}
    cpu = simulate(grid, [obj], source, **kwargs)
    gpu = simulate_cuda(grid, [obj], source, device="cuda:0", dtype="float64", **kwargs)
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    np.testing.assert_allclose(
        gpu.monitor.normalized_far_field(angles),
        cpu.monitor.normalized_far_field(angles),
        rtol=3e-11,
        atol=3e-13,
    )
    assert gpu.diagnostics["backend"] == "compiled_cuda"


def test_unresolved_pec_hole_cuts_are_explicitly_rejected():
    ring = object_from_definition(square_ring_smoke_scene(material_kind="pec")["objects"][0])
    axis_x = np.asarray([0.0, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.2])
    axis_y = np.linspace(0.0, 1.2, 13)
    with pytest.raises(
        ValueError, match="PEC splits an edge|Multiple PEC cuts|Unresolved vacuum gap"
    ):
        CutCellPEC(Grid(axis_x, axis_y), [ring])
