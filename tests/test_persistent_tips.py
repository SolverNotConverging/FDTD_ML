"""Physical fallback size, bounded repairs and hybrid optimization contracts."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh import BoundaryPolicy, DomainPolicy, Mesh, MeshOptions, Simulation
from fdtdmesh.optimization import ReferenceSettings
from fdtdmesh.optimization.feasible import FeasibleSpace
from fdtdmesh.scene import Scene2D
from fdtdmesh.solver.conformal import UnresolvedGeometryError, build_conformal


def test_persistent_tip_patch_shrinks_without_disappearing():
    scene = Scene2D(1, 1).add_polygon([(0.2, 0.8), (0.275, 0.72), (0.22, 0.72)])
    original = scene.as_dict()
    reports = []
    for n in (40, 80):
        mesh = Mesh(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        report = {}
        build_conformal(scene, mesh, boundary=BoundaryPolicy(mode="hybrid"), report=report)
        reports.append(report)
        assert sum(p["cells"] for p in report["patches"]) == report["fallback_cells"]
        assert sum(p["area_m2"] for p in report["patches"]) == pytest.approx(
            report["fallback_area_m2"]
        )
    assert reports[0]["fallback_cells"] == reports[1]["fallback_cells"] > 0
    assert reports[1]["fallback_area_m2"] == pytest.approx(reports[0]["fallback_area_m2"] / 4)
    assert reports[1]["fallback_max_diameter_m"] == pytest.approx(
        reports[0]["fallback_max_diameter_m"] / 2
    )
    assert scene.as_dict() == original
    with pytest.raises(UnresolvedGeometryError, match="diameter"):
        build_conformal(
            scene, mesh, boundary=BoundaryPolicy(mode="hybrid", max_patch_diameter=0.01)
        )


def test_graded_margin_moves_while_exterior_and_clearance_stay_fixed(tmp_path):
    sim = Simulation(
        fmin=0.9e9,
        fmax=1.1e9,
        domain=DomainPolicy(margin_mesh="graded"),
        boundary=BoundaryPolicy(mode="hybrid"),
    )
    sim.add_circle((0, 0), 0.14)
    mesh = sim.apply_mesh("geometry_aware", cells=(100, 100))
    space = FeasibleSpace(sim, mesh)
    side = sim.pml.x.cells + sum(sim.layout.exterior_cells)
    margin = sim.layout.scatterer_margin_cells
    axis = space.axes[0]
    index = side + margin - 1
    assert index in axis.free
    target = mesh.x.copy()
    target[index] += 0.15 * axis.h[index]
    lines = axis.project(target, mesh.x, 0.5)
    assert abs(lines[index] - mesh.x[index]) > 1e-6
    np.testing.assert_array_equal(lines[: side + 1], mesh.x[: side + 1])
    np.testing.assert_array_equal(lines[-side - 1 :], mesh.x[-side - 1 :])
    assert lines[side + margin] == mesh.x[side + margin]
    sim.apply_mesh("custom", mesh=Mesh(lines, mesh.y))
    sim.save(tmp_path / "graded.h5")
    restored = Simulation.load(tmp_path / "graded.h5")
    assert restored.layout.margin_mesh == "graded"
    assert restored.configuration() == sim.configuration()


def test_hybrid_repairs_are_bounded_and_keep_best_physical_patch():
    sim = Simulation(fmin=0.9e9, fmax=1.1e9, boundary=BoundaryPolicy(mode="hybrid"))
    sim.add_rectangle((-0.1, 0.1), (-0.1, 0.1))
    sim.add_rectangle((0.0215, 0.0245), (0.07, 0.2), material="air")
    original = sim.geometry
    mesh = sim.apply_mesh(
        "geometry_aware", cells=(100, 100), options=MeshOptions(hybrid_repair_passes=2)
    )
    report = mesh.metadata["geometry_aware"]
    valid = [p for p in report["passes"] if p["status"] == "valid"]
    assert 1 <= len(valid) <= 3
    best = min((p["fallback_max_diameter_m"], p["fallback_area_m2"]) for p in valid)
    details = sim.discretization.boundary_report
    assert (details["fallback_max_diameter_m"], details["fallback_area_m2"]) == best
    assert sim.geometry is original and (mesh.Nx, mesh.Ny) == (100, 100)
    assert report["termination_reason"]


def test_public_optimizer_defaults_hybrid_and_allows_strict_override(monkeypatch, tmp_path):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle((0, 0), 0.14)
    reference = SimpleNamespace(qualified=True, result=object())
    seen = []
    monkeypatch.setattr(
        module, "_optimize_mesh", lambda candidate, *_a, **_k: seen.append(candidate)
    )
    for policy in (None, BoundaryPolicy(mode="conformal")):
        module.optimize_mesh(
            sim,
            reference,
            cells=(100, 100),
            directory=tmp_path,
            strategy="differential_evolution",
            boundary=policy,
        )
    assert [candidate._boundary.mode for candidate in seen] == ["hybrid", "conformal"]
    assert sim._boundary.mode == "conformal" and sim.mesh is None


def test_hybrid_reference_requires_real_subdivision():
    with pytest.raises(ValueError, match="subdivide"):
        ReferenceSettings(boundary_mode="hybrid")
    assert ReferenceSettings(method="subdivide", boundary_mode="hybrid").boundary_mode == "hybrid"
    with pytest.raises(ValueError):
        MeshOptions(hybrid_repair_passes=-1)


def test_hybrid_reference_keeps_policy_and_checks_patch_localization(tmp_path, monkeypatch):
    """Synthetic fields test qualification control flow, not physical accuracy."""
    from dataclasses import replace

    from test_benchmarks import synthetic

    from fdtdmesh.result import json_text

    module = importlib.import_module("fdtdmesh.optimization.reference")
    sim = Simulation(fmin=0.9e9, fmax=1.1e9, boundary=BoundaryPolicy(mode="hybrid"))
    sim.add_rectangle((-0.1, 0.1), (-0.1, 0.1))
    sim.add_rectangle((0.0215, 0.0245), (0.07, 0.2), material="air")
    sim.apply_mesh("quasi_uniform", cells=(100, 100))
    assert sim.discretization.boundary_report["fallback_cells"] > 0
    seen = []

    def run(candidate, directory):
        seen.append(candidate._boundary.mode)
        result = synthetic(candidate)
        field = np.ones_like(result.far_field)
        result = replace(
            result,
            far_field=field,
            scattering_width=abs(field) ** 2,
            geometry=candidate.computational_geometry,
            _diagnostics_json=json_text(
                dict(result.diagnostics, boundary=candidate.discretization.boundary_report)
            ),
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"synthetic-{len(seen)}.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(module, "run_cached", run)
    reference = module.qualify_reference(
        sim,
        directory=tmp_path,
        settings=ReferenceSettings(method="subdivide", boundary_mode="hybrid", factors=(1, 2, 3)),
    )
    assert reference.qualified
    assert set(seen) == {"hybrid"}
    assert set(reference.report["checks"]) == {
        "temporal",
        "pml",
        "contour",
        "fallback_localization",
    }
    assert reference.report["checks"]["fallback_localization"]["passed"]
    assert reference.report["levels"][0]["boundary"]["fallback_cells"] > 0
