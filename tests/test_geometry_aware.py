"""Exact geometry/domain transforms and deterministic conformal mesh construction."""

import numpy as np
import pytest

from fdtdmesh import DomainPolicy, Geometry, Mesh, Simulation
from fdtdmesh.benchmarks.shapes import make_geometry
from fdtdmesh.geometry_mesher import GeometryMeshingError, inspect_mesh
from fdtdmesh.mesh import MeshInfeasibleError


def test_unbounded_geometry_preserves_coordinates_and_final_csg_bounds(tmp_path):
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_rectangle((-2.0, 2.0), (-1.0, 1.0))
    sim.add_rectangle((0.0, 20.0), (-20.0, 20.0), material="air")
    original = sim.geometry
    assert original.size is None
    assert original.bounds == pytest.approx((-2.0, 0.0, -1.0, 1.0))
    assert np.allclose(
        np.subtract(sim.computational_geometry.bounds, np.repeat(sim.coordinate_offset, 2)),
        original.bounds,
    )
    assert sim.geometry is original
    path = tmp_path / "geometry.json"
    original.save(path)
    assert Geometry.load(path) == original
    assert original.rasterize((8, 8)).shape == (1, 8, 8)
    sim.save(tmp_path / "unmeshed.h5")
    loaded = Simulation.load(tmp_path / "unmeshed.h5")
    assert loaded.geometry == original
    assert loaded.coordinate_offset == sim.coordinate_offset


def test_fixed_exterior_policy_and_roundtrip_after_mesh(tmp_path):
    sim = Simulation(fmin=0.9e9, fmax=1.1e9, domain=DomainPolicy(exterior_spacing=0.01))
    sim.add_circle((-0.03, 0.02), 0.05)
    original = sim.geometry
    first = sim.apply_mesh("geometry_aware")
    layout, size, offset = sim.layout, sim.size, sim.coordinate_offset
    second = sim.apply_mesh("geometry_aware", target_spacing=0.006)
    assert sim.layout == layout and sim.size == size and sim.coordinate_offset == offset
    assert sim.geometry is original
    for a, b in ((first.x, second.x), (first.y, second.y)):
        assert np.allclose(a[:28], b[:28])
        assert np.allclose(a[-28:], b[-28:])
        assert np.allclose(np.diff(b[:28]), 0.01)
    bounds = sim.computational_geometry.bounds
    assert bounds[0] - layout.tfsf_box[0] == pytest.approx(0.05)
    assert layout.tfsf_box[0] - layout.contour_box[0] == pytest.approx(0.04)
    assert layout.contour_box[0] - sim.pml.x.thickness == pytest.approx(0.06)
    assert sim.pml.x.thickness == pytest.approx(0.12)
    sim.save(tmp_path / "prepared.h5")
    loaded = Simulation.load(tmp_path / "prepared.h5")
    assert loaded.geometry == original
    assert np.array_equal(loaded.mesh.x, second.x)
    loaded.add_circle((0.12, 0.02), 0.02)
    assert loaded.mesh is None and loaded.size[0] > size[0]


@pytest.mark.parametrize("case", ["metal", "air", "donor"])
def test_exact_scan_detects_three_rejection_cases(case):
    g = Geometry((4.0, 3.0))
    if case == "metal":
        g, _ = g.added("rectangle", (1.4, 1.6, 0.7, 2.3))
    else:
        g, _ = g.added("rectangle", (0.5, 3.5, 0.7, 2.3))
        lo, hi = (1.4, 1.6) if case == "air" else (1.8, 2.6)
        g, _ = g.added("rectangle", (lo, hi, 0.5, 2.5), "air")
    issues, requests = inspect_mesh(g, Mesh(np.arange(5.0), np.arange(4.0)))
    expected = "unavailable_donor" if case == "donor" else "multiple_crossings"
    assert any(i["kind"] == expected and i["axis"] == 0 and i["fixed"] == 1 for i in issues)
    assert len(requests[0]) >= (3 if case == "donor" else 1)


@pytest.mark.parametrize("shape", ["circle", "ellipse", "triangle", "moon", "star", "wifi"])
def test_catalog_geometry_aware_prepares_valid_conformal_mesh(shape):
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.set_geometry(make_geometry(shape, size=(2.0, 2.0), scale=0.2, incidence_deg=30))
    original = sim.geometry
    mesh = sim.apply_mesh("geometry_aware", time_limit=30)
    assert sim.geometry is original
    assert mesh.metadata["geometry_aware"]["status"] == "valid"
    assert inspect_mesh(sim.computational_geometry, mesh)[0] == []
    assert sim.discretization.dt > 0


def test_limits_are_explicit_and_failure_preserves_existing_mesh():
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle((0.0, 0.0), 0.05)
    mesh = sim.apply_mesh("geometry_aware")
    with pytest.raises(GeometryMeshingError, match="max_cells") as exc:
        sim.apply_mesh("geometry_aware", max_cells=(54, 54))
    assert exc.value.report["max_cells"] == [54, 54]
    assert sim.mesh is mesh
    with pytest.raises(ValueError, match="target_spacing"):
        sim.apply_mesh("geometry_aware", target_spacing=0)
    with pytest.raises(ValueError, match="chooses cell counts"):
        sim.apply_mesh("geometry_aware", cells=(100, 100))
    from fdtdmesh import AxisConstraints

    with pytest.raises(MeshInfeasibleError, match="fixed exterior"):
        sim.apply_mesh("geometry_aware", constraints=AxisConstraints(max_spacing=0.001))


def test_empty_domain_and_outside_phase_origin_are_rejected():
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    with pytest.raises(ValueError, match="nonempty"):
        sim.apply_mesh("geometry_aware")
    sim = Simulation(fmin=0.9e9, fmax=1.1e9, domain=DomainPolicy(phase_origin=(0, 0)))
    sim.add_circle((10.0, 10.0), 0.05)
    with pytest.raises(ValueError, match="phase_origin"):
        sim.apply_mesh("geometry_aware")


def test_irrelevant_air_cutter_does_not_expand_domain_or_prevent_preparation():
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle((0.0, 0.0), 0.05)
    size = sim.size
    sim.add_circle((10.0, 10.0), 0.01, material="air")
    assert sim.size == size
    sim.apply_mesh("geometry_aware")


@pytest.mark.cuda
def test_automatic_domain_gpu_roundtrip_keeps_geometry_and_convergence_status(tmp_path):
    from fdtdmesh import Result
    from fdtdmesh.solver.tmz import cuda_backend

    try:
        cuda_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle((-0.1, 0.2), 0.05)
    sim.apply_mesh("geometry_aware")
    result = sim.solve()
    result.save(tmp_path / "automatic.h5")
    restored = Result.load(tmp_path / "automatic.h5")
    assert restored.converged
    assert restored.input_geometry == sim.geometry
    assert restored.geometry == sim.computational_geometry
    assert restored.diagnostics["geometry_aware"]["status"] == "valid"
