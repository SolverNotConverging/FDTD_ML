import math

import numpy as np
import pytest

import fdtdmesh as fd
from fdtdmesh.benchmarks import SHAPES, make_geometry, make_simulation
from fdtdmesh.scene import Scene2D, ellipse_matrix


def test_axis_aligned_ellipse_intersections_are_exact():
    scene = Scene2D(2.0, 2.0)
    scene.add_ellipse((1.0, 1.0), (0.6, 0.3))

    assert scene.vacuum_intervals(0, 1.0) == pytest.approx([(0.0, 0.4), (1.6, 2.0)])


def test_rotated_ellipse_intersections_match_quadratic_roots():
    centre = np.array([1.0, 1.0])
    radii = (0.6, 0.3)
    angle = 0.37
    fixed = 1.12
    scene = Scene2D(2.0, 2.0)
    scene.add_ellipse(centre, radii, angle)

    q = ellipse_matrix((*centre, *radii, angle))
    offset = fixed - centre[1]
    a, b, c = q[0, 0], 2 * q[0, 1] * offset, q[1, 1] * offset**2 - 1
    roots = np.sort(centre[0] + np.roots((a, b, c)))
    expected = [(0.0, roots[0]), (roots[1], 2.0)]
    assert scene.vacuum_intervals(0, fixed) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["circle", "ellipse"])
def test_tangent_circle_or_ellipse_leaves_full_vacuum_line(kind):
    scene = Scene2D(2.0, 2.0)
    if kind == "circle":
        scene.add_circle((1.0, 1.0), 0.3)
    else:
        scene.add_ellipse((1.0, 1.0), (0.6, 0.3))

    intervals = scene.vacuum_intervals(0, 1.3)
    assert sum(hi - lo for lo, hi in intervals) == pytest.approx(2.0)


def test_rotated_rectangle_near_horizontal_edge_has_stable_intervals():
    scene = Scene2D(2.0, 2.0)
    scene.add_rectangle((0.6, 1.4), (0.7, 1.3))
    rotated = fd.Geometry((2.0, 2.0))
    rotated, _ = rotated.added("rectangle", (0.6, 1.4, 0.7, 1.3))
    rotated = rotated.rotated(1e-12, origin=(1.0, 1.0))

    intervals = rotated.to_scene().vacuum_intervals(0, 1.0)
    assert all(hi > lo for lo, hi in intervals)
    assert sum(hi - lo for lo, hi in intervals) == pytest.approx(1.2, abs=1e-10)


def test_geometry_rotation_preserves_material_recipe_and_inverse_contains():
    geometry = fd.Geometry((4.0, 4.0))
    geometry, _ = geometry.added("circle", (2.0, 2.0, 0.8), "PEC")
    geometry, _ = geometry.added("circle", (2.2, 2.1, 0.35), "air")
    rotated = geometry.rotated(0.73, origin=(2.0, 2.0)).rotated(-0.73, origin=(2.0, 2.0))

    assert [s.material for s in rotated.shapes] == ["PEC", "air"]
    assert [s.kind for s in rotated.shapes] == ["circle", "circle"]
    points = np.array([[2.0, 2.0], [2.65, 2.0], [2.2, 2.1], [0.5, 0.5]])
    assert np.array_equal(
        geometry.contains(points[:, 0], points[:, 1]), rotated.contains(points[:, 0], points[:, 1])
    )


@pytest.mark.parametrize("incidence", [0, 30, 60, 90])
def test_all_catalog_shapes_are_mesh_independent_at_each_incidence(incidence):
    size = (6e-6, 6e-6)
    for name in SHAPES:
        geometry = make_geometry(name, size=size, scale=0.6e-6, incidence_deg=incidence)
        assert geometry.size == size
        assert geometry.shapes
        assert all(shape.material in ("PEC", "air") for shape in geometry.shapes)


@pytest.mark.parametrize("frequency", [0.0, -1.0, math.nan, math.inf])
def test_make_simulation_rejects_nonpositive_or_nonfinite_frequency(frequency):
    with pytest.raises((ValueError, FloatingPointError)):
        make_simulation(frequency=frequency)


def test_wifi_relative_scale_matches_si_geometry_without_raster(monkeypatch):
    def no_raster(*args, **kwargs):
        raise AssertionError("Geometry validation must not use an image")

    monkeypatch.setattr(fd.Geometry, "rasterize", no_raster)
    relative = make_simulation("wifi", scale_factor=1.5)
    physical = make_simulation("wifi", scale=1.5 * 0.6 * relative.wavelength)
    assert relative.geometry == physical.geometry
    assert make_simulation("wifi").geometry == make_simulation("wifi", scale_factor=1).geometry
    relative.apply_mesh("uniform", cells=(192, 192))
    assert relative.mesh.Nx == relative.mesh.Ny == 192


def test_oversized_si_scale_reports_actual_primitive_bounds():
    with pytest.raises(ValueError, match=r"PEC geometry bounds.*domain"):
        make_simulation("wifi", scale=1.5, fit_domain=False)


@pytest.mark.parametrize("factor", [0, -1, math.nan, math.inf])
def test_invalid_relative_scale(factor):
    with pytest.raises(ValueError, match="scale_factor"):
        make_simulation("wifi", scale_factor=factor)


def test_scale_units_cannot_be_mixed():
    with pytest.raises(ValueError, match="not both"):
        make_simulation("wifi", scale=0.1, scale_factor=1.5)
