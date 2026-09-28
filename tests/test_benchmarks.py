"""Study control-flow tests use synthetic fields, never physics qualification."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from legacy_fixtures import make_simulation

from fdtdmesh.mesh import MeshInfeasibleError, adjacent_ratio, axis_mesh
from fdtdmesh.optimization import Reference
from fdtdmesh.optimization.common import errors, experiment_key
from fdtdmesh.optimization.optimize import _optimize_mesh as optimize_mesh
from fdtdmesh.result import Result, json_text
from fdtdmesh.simulation import MeshClearanceError


def synthetic(sim):
    # A deterministic, mesh-dependent objective for checkpoint equivalence.
    v = 1 + np.sum(np.diff(sim.mesh.x) ** 2)
    field = np.full((2, 4), v + 0.1j)
    return Result(
        np.array([0.9e9, 1.1e9]),
        np.arange(4.0),
        field,
        abs(field) ** 2,
        np.empty((0, 8)),
        np.empty((0, 2, 3)),
        sim.geometry,
        sim.mesh,
        json_text(dict(status="converged", dt=1e-12, gpu_ms=0.0, Nt=1)),
        json_text(sim.configuration()),
    )


def test_complex_loss_detects_phase_and_rejects_mismatched_samples():
    common = dict(
        frequencies=np.array([1.0, 2.0]), angles=np.arange(4), converged=True, width=np.ones((2, 4))
    )
    a = SimpleNamespace(**common, far_field=np.ones((2, 4), complex))
    b = SimpleNamespace(**common, far_field=1j * np.ones((2, 4)))
    assert errors(a, b)["error"] == pytest.approx(np.sqrt(2))
    assert errors(a, b)["width_error"] == 0
    b.angles = b.angles + 0.1
    with pytest.raises(ValueError, match="samples"):
        errors(a, b)


@pytest.mark.parametrize("ppw", [48, 108])
def test_moon_reference_aligns_thin_tip_without_changing_geometry_or_budget(ppw):
    from fdtdmesh.optimization.common import refined
    from fdtdmesh.optimization.features import unresolved_edge_anchors

    sim = make_simulation("moon", incidence_deg=30, scale_factor=1.2)
    result = refined(sim, ppw)
    assert result.geometry == sim.geometry
    expected = np.rint(np.array(sim.size) / (sim.wavelength / ppw)).astype(int)
    assert (result.mesh.Nx, result.mesh.Ny) == tuple(expected)
    assert result.mesh.metadata["reference_anchor_passes"] >= 1
    assert not any(len(a) for a in unresolved_edge_anchors(result.geometry, result.mesh))
    for axis in (result.mesh.x, result.mesh.y):
        assert adjacent_ratio(np.diff(axis)) <= 1.4 + 1e-8


def test_reference_reports_every_preparation_failure(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.optimization.reference")
    from fdtdmesh.solver.conformal import UnresolvedGeometryError

    def rejected(*args):
        raise UnresolvedGeometryError("unresolved test edge")

    monkeypatch.setattr(module, "refined", rejected)
    updates = []
    reference = module.qualify_reference(
        make_simulation(), directory=tmp_path, progress=updates.append
    )
    assert reference.report["status"] == "no_valid_reference"
    assert not reference.qualified and reference.result is None
    assert len(updates) == len(module.ReferenceSettings().ppw)
    assert all(item["message"] == "unresolved test edge" for item in updates)


def test_local_and_fixed_anchor_projection_enforce_hard_constraints():
    for kwargs in ({"anchor_window": 2}, {"fixed_indices": {12: 0.31, 27: 0.67}}):
        x = axis_mesh(1, 40, [1, 2, 1], [0.31, 0.67], **kwargs)
        assert len(x) == 41 and 0.31 in x and 0.67 in x
        assert adjacent_ratio(np.diff(x)) <= 1.4 + 1e-8
        if "fixed_indices" in kwargs:
            assert x[12] == 0.31 and x[27] == 0.67


def test_de_resume_matches_uninterrupted_and_preserves_input(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = make_simulation()
    sim._apply_mesh("uniform", cells=(144, 144))
    original = sim.mesh
    ref = Reference(
        synthetic(sim),
        dict(
            qualified=True,
            case=experiment_key(sim),
            key="synthetic-test-only",
            settings=dict(rtol=0.002, worst_rtol=0.005),
        ),
        tmp_path,
    )

    def fake_run(candidate, directory):
        result = synthetic(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "synthetic.h5"
        # Unique result paths preserve earlier winning candidates.
        from fdtdmesh.strategies import mesh_id

        path = directory / f"{mesh_id(candidate.mesh)}.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(module, "run_cached", fake_run)
    kwargs = dict(cells=(144, 144), population=4, controls=3, max_seconds=120)
    full = optimize_mesh(sim, ref, directory=tmp_path / "full", max_evaluations=10, **kwargs)

    def interrupt_after_seven(trial):
        if trial["number"] == 7:
            raise RuntimeError("test interruption after checkpoint")

    with pytest.raises(RuntimeError, match="test interruption"):
        optimize_mesh(
            sim,
            ref,
            directory=tmp_path / "split",
            max_evaluations=10,
            progress=interrupt_after_seven,
            **kwargs,
        )
    resumed = optimize_mesh(sim, ref, directory=tmp_path / "split", max_evaluations=10, **kwargs)
    for key in ("population", "energies", "rng_state", "best_error", "index", "generation"):
        assert resumed.report[key] == full.report[key]
    assert [t["parameters"] for t in resumed.report["trials"]] == [
        t["parameters"] for t in full.report["trials"]
    ]
    assert sim.mesh is original
    archive = (resumed.directory / "optimization.h5").read_bytes()
    changed = {**kwargs, "cells": (80, 80)}
    other = optimize_mesh(sim, ref, directory=tmp_path / "split", max_evaluations=10, **changed)
    assert other.directory != resumed.directory
    assert (resumed.directory / "optimization.h5").read_bytes() == archive
    import json

    description = json.loads((other.directory / "experiment.json").read_text())
    assert description["optimizer"]["cells"] == [80, 80]
    assert description["simulation"]["geometry"] == json.loads(json_text(sim.geometry.as_dict()))
    assert description["optimizer"]["max_evaluations"] == 10
    ref.report["qualified"] = False
    with pytest.raises(ValueError, match="qualified"):
        optimize_mesh(sim, ref, directory=tmp_path / "bad", **kwargs)


def test_geometry_seed_optimizes_at_same_budget_with_witness_anchors(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = make_simulation("star", incidence_deg=30, scale_factor=0.5)
    baseline = sim.apply_mesh("geometry_aware")
    original = sim.mesh
    ref = Reference(
        synthetic(sim),
        dict(
            qualified=True,
            case=experiment_key(sim),
            key="synthetic-seeded",
            settings=dict(rtol=0.002, worst_rtol=0.005),
        ),
        tmp_path,
    )
    meshes = []

    def fake_run(candidate, directory):
        meshes.append(candidate.mesh)
        result = synthetic(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        from fdtdmesh.strategies import mesh_id

        path = directory / f"{mesh_id(candidate.mesh)}.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(module, "run_cached", fake_run)
    out = optimize_mesh(
        sim,
        ref,
        cells=(baseline.Nx, baseline.Ny),
        initial_mesh=baseline,
        directory=tmp_path / "seeded",
        max_evaluations=6,
        population=4,
        controls=3,
    )
    assert [t["kind"] for t in out.report["trials"][:3]] == [
        "geometry_aware",
        "uniform",
        "deterministic",
    ]
    assert meshes[0].x.tobytes() == baseline.x.tobytes()
    assert meshes[0].y.tobytes() == baseline.y.tobytes()
    assert sim.mesh is original
    for mesh in meshes:
        assert (mesh.Nx, mesh.Ny) == (baseline.Nx, baseline.Ny)
        for axis, witnesses in zip(
            (mesh.x, mesh.y), baseline.metadata["geometry_aware"]["witness_anchors"]
        ):
            assert all(np.any(np.isclose(axis, v, atol=1e-12, rtol=0)) for v in witnesses)
        for axis, original_axis in ((mesh.x, baseline.x), (mesh.y, baseline.y)):
            assert np.array_equal(axis[:28], original_axis[:28])
            assert np.array_equal(axis[-28:], original_axis[-28:])
    assert (
        out.report["description"]["optimizer"]["initial_mesh"]["witness_anchors"]
        == (baseline.metadata["geometry_aware"]["witness_anchors"])
    )
    with pytest.raises(ValueError, match="must match"):
        optimize_mesh(
            sim,
            ref,
            cells=(baseline.Nx + 1, baseline.Ny),
            initial_mesh=baseline,
            directory=tmp_path / "invalid",
        )


@pytest.mark.parametrize("check_boundaries", [False, True])
def test_reference_requires_spatial_and_all_sensitivity_checks(
    tmp_path, monkeypatch, check_boundaries
):
    module = importlib.import_module("fdtdmesh.optimization.reference")
    sim = make_simulation()
    sim._apply_mesh("uniform", cells=(144, 144))
    monkeypatch.setattr(module, "refined", lambda *args: sim)
    monkeypatch.setattr(module, "_expanded_pml", lambda *args: sim)

    def fake_run(candidate, directory):
        result = synthetic(candidate)
        path = directory / "synthetic.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(module, "run_cached", fake_run)
    ref = module.qualify_reference(
        sim,
        directory=tmp_path,
        settings=module.ReferenceSettings(ppw=(48, 72, 108), check_boundaries=check_boundaries),
    )
    assert len(ref.report["levels"]) == 3
    assert ref.qualified == check_boundaries
    assert set(ref.report["checks"]) == (
        {"temporal", "pml", "contour"} if check_boundaries else {"temporal"}
    )


def test_tfsf_clearance_is_mesh_infeasibility_and_failed_apply_is_transactional():
    sim = make_simulation(fit_domain=False)
    sim.add_circle(
        (sim.layout.tfsf_box[0] + 0.06 * sim.wavelength, 3 * sim.wavelength), 0.05 * sim.wavelength
    )
    with pytest.raises(MeshClearanceError, match="clearance inside TFSF") as exc:
        sim._apply_mesh("uniform", cells=(144, 144))
    assert isinstance(exc.value, MeshInfeasibleError)
    assert sim.mesh is None


@pytest.mark.parametrize("strategy", ["differential_evolution"])
def test_search_records_clearance_failures_and_continues(tmp_path, monkeypatch, strategy):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = make_simulation()
    sim._apply_mesh("uniform", cells=(144, 144))
    ref = Reference(
        synthetic(sim), dict(qualified=True, case=experiment_key(sim), key="test"), tmp_path
    )

    class RejectedCandidate:
        def _apply_mesh(self, *args, **kwargs):
            raise MeshClearanceError("Geometry and enlarged cells need clearance inside TFSF")

    monkeypatch.setattr(module, "clone", lambda *args, **kwargs: RejectedCandidate())
    out = optimize_mesh(
        sim,
        ref,
        cells=(144, 144),
        directory=tmp_path / strategy,
        strategy=strategy,
        max_evaluations=4,
        max_seconds=60,
    )
    assert len(out.report["trials"]) == 4
    assert out.best is None and out.report["status"] == "no_feasible_mesh"
    assert all(
        t["status"] == "MeshClearanceError" and t["error"] is None for t in out.report["trials"]
    )


def test_search_does_not_swallow_unrelated_value_errors(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = make_simulation()
    sim._apply_mesh("uniform", cells=(144, 144))
    ref = Reference(
        synthetic(sim), dict(qualified=True, case=experiment_key(sim), key="test"), tmp_path
    )

    class BrokenCandidate:
        def _apply_mesh(self, *args, **kwargs):
            raise ValueError("unexpected programming error")

    monkeypatch.setattr(module, "clone", lambda *args, **kwargs: BrokenCandidate())
    with pytest.raises(ValueError, match="unexpected programming error"):
        optimize_mesh(sim, ref, cells=(144, 144), directory=tmp_path / "broken")


@pytest.mark.parametrize("strategy", ["differential_evolution"])
@pytest.mark.parametrize("anchored", [False, True])
def test_optional_anchors_apply_to_baselines_and_search(tmp_path, monkeypatch, strategy, anchored):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    from fdtdmesh.solver.conformal import UnresolvedGeometryError

    sim = make_simulation()
    sim._apply_mesh("uniform", cells=(144, 144))
    ref = Reference(
        synthetic(sim), dict(qualified=True, case=experiment_key(sim), key="test"), tmp_path
    )
    feature_calls, proposals = [], []
    anchors = (np.array([0.4]), np.array([0.5]))

    def features(geometry):
        feature_calls.append(geometry)
        return anchors

    class RejectedCandidate:
        def _apply_mesh(self, strategy, **kwargs):
            proposals.append((strategy, kwargs))
            raise UnresolvedGeometryError("test unresolved crossing")

    monkeypatch.setattr(module, "geometry_anchors", features)
    monkeypatch.setattr(module, "clone", lambda *args, **kwargs: RejectedCandidate())
    # Omit the option in the False case to exercise the public default.
    options = {"feature_anchors": True} if anchored else {}
    out = optimize_mesh(
        sim,
        ref,
        cells=(144, 144),
        directory=tmp_path,
        strategy=strategy,
        max_evaluations=4,
        **options,
    )
    assert len(feature_calls) == int(anchored)
    assert [p[0] for p in proposals[:2]] == ["uniform", "deterministic"]
    assert all(p[0] == "density" for p in proposals[2:])
    assert len(proposals) == 4
    assert all(p[1]["anchors"] is (anchors if anchored else None) for p in proposals)
    assert all(p[1]["cells"] == (144, 144) for p in proposals)
    assert out.report["description"]["optimizer"]["feature_anchors"] is anchored
    assert out.best is None and out.report["status"] == "no_feasible_mesh"
    assert all(t["status"] == "UnresolvedGeometryError" for t in out.report["trials"])
