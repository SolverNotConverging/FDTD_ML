"""Feasible-search invariants and checkpoint control flow; synthetic loss only."""

import importlib

import numpy as np
import pytest
from legacy_fixtures import make_simulation
from test_benchmarks import synthetic

from fdtdmesh import Simulation
from fdtdmesh.catalog import make_engineered_geometry
from fdtdmesh.geometry_mesher import inspect_mesh
from fdtdmesh.mesh import validate_spacing
from fdtdmesh.optimization import (
    FeasibleSettings,
    Reference,
    analyze_mesh_adaptivity,
)
from fdtdmesh.optimization.common import experiment_key
from fdtdmesh.optimization.feasible import FeasibleSpace
from fdtdmesh.optimization.optimize import _optimize_mesh as optimize_mesh
from fdtdmesh.strategies import mesh_id


def test_local_proposals_restore_aircraft_failures_without_losing_variation():
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.set_geometry(make_engineered_geometry("swept_aircraft", scale=1))
    seed = sim._apply_mesh("geometry_aware", time_limit=40)
    space = FeasibleSpace(sim, seed)
    accepted, repaired = [], []
    rng = np.random.default_rng(0)
    for _ in range(8):
        mesh, info = space.propose(seed, rng, 0.75, 6, FeasibleSettings())
        if mesh is None:
            continue
        accepted.append(mesh_id(mesh))
        repaired.append(info["accepted_stage"] != "raw")
        assert info["movement"]["max_cells"] >= 0.03
        assert not inspect_mesh(sim.computational_geometry, mesh)[0]
        for original, axis, state in zip((seed.x, seed.y), (mesh.x, mesh.y), space.axes):
            assert len(axis) == len(original)
            assert np.array_equal(axis[state.fixed], original[state.fixed])
            validate_spacing(axis, space.constraints)
    assert len(set(accepted)) >= 4
    assert any(repaired)
    assert sim.mesh is seed


def test_linear_adaptivity_ranges_contain_seed_and_fixed_lines_have_zero_width():
    sim = make_simulation("circle", scale_factor=0.5)
    seed = sim.apply_mesh("geometry_aware")
    report = analyze_mesh_adaptivity(sim, seed)
    assert not report["geometry_constrained"]
    for axis, state in zip((seed.x, seed.y), report["axes"]):
        assert np.all(np.array(state["lower"]) <= axis + 1e-10)
        assert np.all(np.array(state["upper"]) >= axis - 1e-10)
        assert np.all(np.array(state["width_cells"])[state["fixed_indices"]] == 0)
        assert state["mobile_lines"] > 0


def test_feasible_resume_matches_uninterrupted_and_limits_unique_solves(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.optimization.optimize")
    sim = make_simulation("circle", scale_factor=0.5)
    seed = sim.apply_mesh("geometry_aware")
    ref = Reference(
        synthetic(sim),
        dict(
            qualified=True,
            case=experiment_key(sim),
            key="synthetic-local-only",
            settings=dict(rtol=0.002, worst_rtol=0.005),
        ),
        tmp_path,
    )

    def fake_run(candidate, directory):
        result = synthetic(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{mesh_id(candidate.mesh)}.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(module, "run_cached", fake_run)
    kwargs = dict(
        cells=(seed.Nx, seed.Ny),
        initial_mesh=seed,
        strategy="feasible_local",
        max_evaluations=10,
        max_seconds=120,
        population=4,
        controls=3,
    )
    full = optimize_mesh(sim, ref, directory=tmp_path / "full", **kwargs)

    def interrupt(item):
        if item["number"] == 6:
            raise RuntimeError("checkpoint interruption")

    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        optimize_mesh(sim, ref, directory=tmp_path / "split", progress=interrupt, **kwargs)
    resumed = optimize_mesh(sim, ref, directory=tmp_path / "split", **kwargs)
    for key in ("rng_state", "feasible_state", "best_error", "solve_meshes"):
        assert full.report[key] == resumed.report[key]
    assert sim.mesh is seed
    assert len(full.report["solve_meshes"]) >= 6
    limited = optimize_mesh(sim, ref, directory=tmp_path / "limit", max_solves=4, **kwargs)
    assert limited.report["status"] == "solve_limit"
    assert len(limited.report["solve_meshes"]) == 4
    assert limited.report["validation"]["passed"]


@pytest.mark.parametrize(
    "kwargs", [dict(radius=0), dict(min_movement=0.1), dict(exploration=1.1), dict(backtracks=-1)]
)
def test_feasible_settings_validate_before_work(kwargs):
    with pytest.raises(ValueError):
        FeasibleSettings(**kwargs)
