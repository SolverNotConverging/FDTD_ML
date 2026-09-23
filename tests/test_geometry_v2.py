"""Independent geometric checks for the second scattering curriculum."""

import numpy as np
import pytest

from scattermesh import PEC, Ellipse, Grid, Material, Polygon, SmoothLobed, oriented_rectangle
from scattermesh.conformal import CutCellPEC
from scattermesh.curriculum_v2 import generate_lineages, object_from_scene
from scattermesh.geometry import average_materials


def test_rotated_ellipse_matches_independent_quadratic_line_roots():
    ellipse = Ellipse((0.61, 0.59), (0.14, 0.075), 0.43, PEC())
    for axis, fixed in ((0, 0.61), (1, 0.57)):
        intervals = ellipse.line_intervals(fixed, axis)
        assert len(intervals) == 1
        left, right = intervals[0]
        assert right > left
        for p in (left, right):
            x, y = (p, fixed) if axis == 0 else (fixed, p)
            c, s = np.cos(ellipse.angle), np.sin(ellipse.angle)
            u = c * (x - ellipse.center[0]) + s * (y - ellipse.center[1])
            v = -s * (x - ellipse.center[0]) + c * (y - ellipse.center[1])
            assert (u / ellipse.radii[0]) ** 2 + (v / ellipse.radii[1]) ** 2 == pytest.approx(
                1, abs=1e-12
            )


def test_polygon_line_intervals_respect_concave_notch_and_exact_edges():
    shape = Polygon(
        (
            (0.3, 0.3),
            (0.8, 0.3),
            (0.8, 0.8),
            (0.6, 0.8),
            (0.6, 0.5),
            (0.5, 0.5),
            (0.5, 0.8),
            (0.3, 0.8),
        ),
        PEC(),
    )
    assert shape.line_intervals(0.7, 0) == [(0.3, 0.5), (0.6, 0.8)]
    assert shape.contains(0.4, 0.7)
    assert not shape.contains(0.55, 0.7)
    assert shape.contains(0.5, 0.7)
    with pytest.raises(ValueError, match="self-intersect"):
        Polygon(((0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)), PEC())


def test_lobed_intersections_follow_same_continuous_membership():
    shape = SmoothLobed((0.6, 0.6), (0.11, 0.13, 0.12, 0.09, 0.12, 0.13, 0.11, 0.10), 0.23, PEC())
    for axis in (0, 1):
        for fixed in (0.56, 0.6, 0.64):
            intervals = shape.line_intervals(fixed, axis)
            assert intervals
            for left, right in intervals:
                midpoint = (left + right) / 2
                x, y = (midpoint, fixed) if axis == 0 else (fixed, midpoint)
                assert shape.contains(x, y)
                for endpoint in (left, right):
                    a, b = (endpoint, fixed) if axis == 0 else (fixed, endpoint)
                    theta = np.arctan2(b - shape.center[1], a - shape.center[0]) - shape.angle
                    radius = np.hypot(a - shape.center[0], b - shape.center[1])
                    assert radius == pytest.approx(shape._radius(theta), abs=1e-11)


def test_new_shapes_reach_material_and_cut_cell_operators():
    axis = np.linspace(0, 1.2, 65)
    grid = Grid(axis, axis)
    dielectric = oriented_rectangle((0.6, 0.6), 0.22, 0.14, 0.38, Material(4))
    eps, sigma = average_materials(grid, [dielectric], samples=4)
    assert eps.max() == 4
    assert np.all(sigma == 0)
    assert np.count_nonzero(eps > 1) > 0
    pec = CutCellPEC(grid, [Ellipse((0.6, 0.6), (0.13, 0.08), 0.4, PEC())])
    assert pec.boundary_edge_count > 0
    assert 0 < pec.minimum_fraction < 1


def test_conic_tangency_and_unsupported_pec_split_edge():
    ellipse = Ellipse((0.6, 0.6), (0.15, 0.07), 0.31, PEC())
    tangent = ellipse.line_intervals(ellipse.bounds[3], 0)
    assert len(tangent) == 1
    assert tangent[0][1] - tangent[0][0] < 1e-7
    assert ellipse.line_intervals(ellipse.bounds[3] + 1e-4, 0) == []

    scene = next(row for row in generate_lineages(128) if row["lineage_id"] == "v2_circle_015")
    axis = np.linspace(0, 1.2, 33)
    with pytest.raises(ValueError, match="split|unresolved"):
        CutCellPEC(Grid(axis, axis), [object_from_scene(scene)])
