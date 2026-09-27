"""Study control-flow tests use synthetic fields, never physics qualification."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.benchmarks import Reference, make_simulation, optimize_mesh
from fdtdmesh.benchmarks.common import errors, experiment_key
from fdtdmesh.mesh import adjacent_ratio, axis_mesh
from fdtdmesh.result import Result, json_text


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


def test_local_and_fixed_anchor_projection_enforce_hard_constraints():
    for kwargs in ({"anchor_window": 2}, {"fixed_indices": {12: 0.31, 27: 0.67}}):
        x = axis_mesh(1, 40, [1, 2, 1], [0.31, 0.67], **kwargs)
        assert len(x) == 41 and 0.31 in x and 0.67 in x
        assert adjacent_ratio(np.diff(x)) <= 1.4 + 1e-8
        if "fixed_indices" in kwargs:
            assert x[12] == 0.31 and x[27] == 0.67


def test_de_resume_matches_uninterrupted_and_preserves_input(tmp_path, monkeypatch):
    module = importlib.import_module("fdtdmesh.benchmarks.optimize")
    sim = make_simulation()
    sim.apply_mesh("uniform", cells=(144, 144))
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
    optimize_mesh(sim, ref, directory=tmp_path / "split", max_evaluations=7, **kwargs)
    resumed = optimize_mesh(sim, ref, directory=tmp_path / "split", max_evaluations=10, **kwargs)
    for key in ("population", "energies", "rng_state", "best_error", "index", "generation"):
        assert resumed.report[key] == full.report[key]
    assert [t["parameters"] for t in resumed.report["trials"]] == [
        t["parameters"] for t in full.report["trials"]
    ]
    assert sim.mesh is original
    ref.report["qualified"] = False
    with pytest.raises(ValueError, match="qualified"):
        optimize_mesh(sim, ref, directory=tmp_path / "bad", **kwargs)


@pytest.mark.parametrize("check_boundaries", [False, True])
def test_reference_requires_spatial_and_all_sensitivity_checks(
    tmp_path, monkeypatch, check_boundaries
):
    module = importlib.import_module("fdtdmesh.benchmarks.reference")
    sim = make_simulation()
    sim.apply_mesh("uniform", cells=(144, 144))
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
