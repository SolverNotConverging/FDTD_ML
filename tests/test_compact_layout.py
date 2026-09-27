"""Exact CSG bounds, reserved exterior cells, and fully refined references."""

import numpy as np
import pytest

from fdtdmesh import Geometry, Simulation
from fdtdmesh.benchmarks import make_simulation
from fdtdmesh.benchmarks.common import refined
from fdtdmesh.mesh import MeshInfeasibleError
from fdtdmesh.scattering import box_indices, grid_index


def test_wifi_bounds_follow_visible_arcs_not_air_cutters():
    sim = make_simulation("wifi", scale_factor=1.7)
    bounds = sim.geometry.bounds
    scale = 0.6 * 1.7 * sim.wavelength
    assert bounds[1] - bounds[0] == pytest.approx(2 * np.sqrt(0.95**2 - 0.2**2) * scale)
    assert bounds[3] - bounds[2] == pytest.approx((0.65 + 0.57) * scale)
    # The cutter is larger than the PEC material and is allowed inside PML.
    assert min(b[2] for b in sim.geometry.to_scene().bounds()) < bounds[2]


def test_bounds_of_clipped_ellipse_and_hidden_outside_cutter():
    geometry = Geometry((10, 10))
    geometry, _ = geometry.added("ellipse", (5, 5, 3, 2, 0))
    geometry, _ = geometry.added("rectangle", (5, 12, -1, 11), "air")
    assert geometry.bounds == pytest.approx((2, 5, 3, 7))
    # Rotation and translation preserve physical extents with no grid involved.
    rotated = geometry.rotated(np.pi / 2)
    assert rotated.bounds == pytest.approx((3, 7, 2, 5))


def test_compact_allocation_and_fixed_exterior_across_strategies(tmp_path):
    sim = make_simulation("wifi", scale_factor=1.7)
    meshes = []
    for strategy, kwargs in [
        ("uniform", {}),
        ("deterministic", {}),
        ("density", dict(density=([1, 1.3, 1], [1.2, 1, 1.1]))),
    ]:
        mesh = sim.apply_mesh(strategy, cells=(192, 192), **kwargs)
        a, b, c, d = box_indices(mesh, sim.layout.tfsf_box)
        ca, cb, cc, cd = box_indices(mesh, sim.layout.contour_box)
        assert (a, b, c, d) == (22, 170, 22, 170)
        assert (ca, cb, cc, cd) == (18, 174, 18, 174)
        for lines, low, high in (
            (mesh.x, *sim.geometry.bounds[:2]),
            (mesh.y, *sim.geometry.bounds[2:]),
        ):
            assert grid_index(lines, low) == 27 and grid_index(lines, high) == 165
        meshes.append(mesh)
    for mesh in meshes[1:]:
        for axis in ("x", "y"):
            np.testing.assert_array_equal(getattr(mesh, axis)[:28], getattr(meshes[0], axis)[:28])
            np.testing.assert_array_equal(getattr(mesh, axis)[165:], getattr(meshes[0], axis)[165:])
    assert not np.array_equal(meshes[0].x[28:165], meshes[1].x[28:165])
    sim.save(tmp_path / "compact.h5")
    loaded = Simulation.load(tmp_path / "compact.h5")
    assert loaded.layout == sim.layout
    np.testing.assert_array_equal(loaded.mesh.x, sim.mesh.x)
    with pytest.raises(MeshInfeasibleError):
        sim.apply_mesh("uniform", cells=(48, 48))
    assert sim.mesh is meshes[-1]


def test_reference_refines_exterior_without_moving_boxes():
    sim = make_simulation()
    sim.apply_mesh("uniform", cells=(192, 192))
    fine = refined(sim, 72)
    assert fine.layout.contour_box == sim.layout.contour_box
    assert fine.layout.tfsf_box == sim.layout.tfsf_box
    assert fine.layout.exterior_cells is None
    assert fine.layout.scatterer_margin_cells is None
    ca = grid_index(fine.mesh.x, fine.layout.contour_box[0])
    a = grid_index(fine.mesh.x, fine.layout.tfsf_box[0])
    assert ca - fine.pml.x.cells > 6 and a - ca > 4
