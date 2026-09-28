"""Breaking public API, wavelength layout and mixed-operator numerical gates."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from fdtdmesh import BoundaryPolicy, DomainPolicy, Simulation
from fdtdmesh.mesh import Mesh
from fdtdmesh.scene import Scene2D
from fdtdmesh.solver.conformal import UnresolvedGeometryError, build_conformal, stiffness


def test_domain_physical_distances_and_frozen_exterior(tmp_path):
    policy = DomainPolicy()
    allocation = policy.allocation(0.9e9, 1.1e9)
    actual, requested = np.array(allocation["achieved"]), np.array(allocation["requested"])
    assert np.all(actual >= requested - 1e-15)
    assert np.all(actual - requested < allocation["spacing"] + 1e-15)
    sim = Simulation(fmin=0.9e9, fmax=1.1e9, domain=policy)
    sim.add_circle((-0.3, 0.2), 0.14)
    original = sim.geometry
    first = sim.apply_mesh("geometry_aware", cells=(100, 100))
    layout = sim.layout
    sim.apply_mesh("geometry_aware", cells=(120, 120))
    n = sim.pml.x.cells + sum(sim.layout.exterior_cells)
    np.testing.assert_array_equal(first.x[: n + 1], sim.mesh.x[: n + 1])
    assert sim.layout == layout and sim.geometry is original
    assert sim.mesh.Nx == sim.mesh.Ny == 120
    sim.save(tmp_path / "case.h5")
    loaded = Simulation.load(tmp_path / "case.h5")
    assert loaded.geometry == sim.geometry
    assert loaded.configuration() == sim.configuration()
    with pytest.raises(TypeError):
        Simulation(size=(1, 1), fmin=0.9e9, fmax=1.1e9)


@pytest.mark.parametrize("kind", ["island", "hole", "strip", "gap", "three_crossings"])
def test_unsupported_features_fallback_and_positive_operator(kind):
    mesh = Mesh(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    scene = Scene2D(1, 1)
    if kind in ("hole", "gap"):
        scene.add_rectangle((0.2, 0.8), (0.2, 0.8))
    if kind == "island":
        scene.add_circle((0.45, 0.45), 0.02)
    elif kind == "hole":
        scene.add_circle((0.45, 0.45), 0.02, pec=False)
    elif kind == "strip":
        scene.add_rectangle((0.44, 0.46), (0.25, 0.75))
    elif kind == "gap":
        scene.add_rectangle((0.44, 0.46), (0.25, 0.85), pec=False)
    else:
        scene.add_rectangle((0.41, 0.43), (0.3, 0.7))
        scene.add_rectangle((0.46, 0.65), (0.3, 0.7))
    before = scene.contains(mesh.x[:, None], mesh.y[None, :])
    with pytest.raises(UnresolvedGeometryError):
        build_conformal(scene, mesh)
    report = {}
    pec, hx, hy, bound = build_conformal(
        scene, mesh, boundary=BoundaryPolicy(mode="hybrid"), report=report
    )
    assert 0 < report["fallback_cells"] < mesh.Nx * mesh.Ny
    np.testing.assert_array_equal(pec[1:-1, 1:-1], before[1:-1, 1:-1])
    K, mass, _ = stiffness(mesh, pec, hx, hy)
    A = K.toarray() / np.sqrt(mass[:, None] * mass[None, :])
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    eigen = eigvalsh(A)
    assert eigen.min() >= -1e-10 and eigen.max() <= bound * (1 + 1e-12)
    with pytest.raises(UnresolvedGeometryError, match="policy"):
        build_conformal(
            scene, mesh, boundary=BoundaryPolicy(mode="hybrid", max_fallback_fraction=0)
        )


def test_supported_cuts_unchanged_and_hidden_overlay_not_rejected():
    mesh = Mesh(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    scene = Scene2D(1, 1).add_circle((0.5, 0.5), 0.213)
    strict = build_conformal(scene, mesh)
    report = {}
    hybrid = build_conformal(scene, mesh, boundary=BoundaryPolicy(mode="hybrid"), report=report)
    assert report["fallback_cells"] == 0
    for a, b in zip(strict[1:3], hybrid[1:3]):
        np.testing.assert_array_equal(a.diagonal, b.diagonal)
        np.testing.assert_array_equal(a.peer, b.peer)
    overlay = Scene2D(1, 1).add_circle((0.425, 0.425), 0.002)
    overlay.add_rectangle((0.3, 0.7), (0.3, 0.7), pec=False)
    assert not build_conformal(overlay, mesh)[0][1:-1, 1:-1].any()


def test_hybrid_search_uses_custom_seed_and_retains_movement():
    from fdtdmesh import Geometry
    from fdtdmesh.optimization import FeasibleSettings
    from fdtdmesh.optimization.feasible import FeasibleSpace

    sim = Simulation(fmin=0.9e9, fmax=1.1e9, boundary=BoundaryPolicy(mode="hybrid"))
    geometry = Geometry().added("rectangle", (-0.1, 0.1, -0.1, 0.1))[0]
    geometry = geometry.added("rectangle", (0.0215, 0.0245, 0.07, 0.2), "air")[0]
    sim.set_geometry(geometry)
    seed = sim.apply_mesh("quasi_uniform", cells=(100, 100))
    assert "geometry_aware" not in seed.metadata
    space = FeasibleSpace(sim, seed)
    candidate, info = space.propose(seed, np.random.default_rng(2), 0.5, 4, FeasibleSettings())
    assert candidate is not None and (candidate.Nx, candidate.Ny) == (100, 100)
    assert info["movement"]["max_cells"] >= 0.03
    assert sim.geometry is geometry
    for original, changed, axis in zip((seed.x, seed.y), (candidate.x, candidate.y), space.axes):
        np.testing.assert_array_equal(original[axis.fixed], changed[axis.fixed])


def test_public_mesh_contracts_reject_conflicting_options():
    from fdtdmesh import MeshOptions

    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.add_circle((0, 0), 0.14)
    with pytest.raises(ValueError, match="Choose exact"):
        sim.apply_mesh("geometry_aware", cells=(100, 100), target_spacing=0.01)
    with pytest.raises(ValueError, match="own feature anchors"):
        sim.apply_mesh("geometry_aware", options=MeshOptions(anchors=([0.4], [0.4])))
    with pytest.raises(ValueError, match="Unknown"):
        sim.apply_mesh("cnn", cells=(100, 100))


def test_public_optimizer_matches_strict_reference_to_hybrid_case(tmp_path, monkeypatch):
    """Synthetic objective only: verify policy matching and public settings wiring."""
    import importlib

    from test_benchmarks import synthetic

    from fdtdmesh.optimization import Reference, SearchSettings, optimize_mesh
    from fdtdmesh.optimization.common import clone, experiment_key, physical_key
    from fdtdmesh.strategies import mesh_id

    sim = Simulation(fmin=0.9e9, fmax=1.1e9, boundary=BoundaryPolicy(mode="hybrid"))
    sim.add_circle((0, 0), 0.14)
    seed = sim.apply_mesh("quasi_uniform", cells=(100, 100))
    strict = clone(sim)
    strict._boundary = BoundaryPolicy()
    strict.apply_mesh("custom", mesh=seed)

    def snapshot(candidate):
        return replace(synthetic(candidate), geometry=candidate.computational_geometry)

    reference = Reference(
        snapshot(strict),
        dict(
            qualified=True,
            key="synthetic-policy-test",
            case=experiment_key(strict),
            physical_case=physical_key(strict),
            settings=dict(rtol=0.002, worst_rtol=0.005),
        ),
        tmp_path,
    )

    def fake_run(candidate, directory):
        result = snapshot(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{mesh_id(candidate.mesh)}.h5"
        result.save(path)
        return result, path, False

    monkeypatch.setattr(
        importlib.import_module("fdtdmesh.optimization.optimize"), "run_cached", fake_run
    )
    optimized = optimize_mesh(
        sim,
        reference,
        cells=(100, 100),
        directory=tmp_path / "search",
        initial_mesh=seed,
        settings=SearchSettings(max_evaluations=4, population=4, validate_winner=False),
    )
    assert optimized.best is not None
    assert len(optimized.report["trials"]) == 4
    assert "validation" not in optimized.report
    assert sim.mesh is seed


@pytest.mark.cuda
def test_hybrid_native_cuda_matches_cpu_oracle():
    from fdtdmesh.cases import canonical_case
    from fdtdmesh.simulation import prepare
    from fdtdmesh.solver.reference_tmz import run_reference
    from fdtdmesh.solver.tmz import cuda_backend

    try:
        runtime = cuda_backend()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    case, mesh = canonical_case(ppw=16)
    # Add an unresolved remote PEC island inside TFSF, outside the cylinder.
    case.scene.add_circle(
        ((mesh.x[29] + mesh.x[30]) / 2, (mesh.y[48] + mesh.y[49]) / 2),
        0.1 * (mesh.x[30] - mesh.x[29]),
    )
    case = replace(case, boundary=BoundaryPolicy(mode="hybrid"))
    c, box, source, contour, options = prepare(case, mesh)
    assert c.boundary_report["fallback_cells"] > 0
    nt = 173
    options.update(max_steps=nt, check_interval=64, auto_stop=False, debug=True)
    gpu = runtime.run(c, (mesh.Nx, mesh.Ny), box, source, contour, options)
    u = np.arange(1, nt + 1) * c.dt * case.frequency - options["pulse_delay"]
    wave = (
        options["source_scale"]
        * np.exp(-((u / options["pulse_width"]) ** 2))
        * np.cos(2 * np.pi * u)
    )
    cpu = run_reference(c, (mesh.Nx, mesh.Ny), box, wave, source, contour, case.frequency)
    for key, expected in zip(("Ez", "Hx", "Hy", "currents", "incident"), cpu[:5]):
        actual = gpu[4][key]
        if actual.ndim == expected.ndim + 1:
            actual = actual[0]
        assert np.linalg.norm(actual - expected) <= 2e-13 * max(np.linalg.norm(expected), 1e-30)
