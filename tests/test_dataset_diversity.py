from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.generate import (
    GenerationConfig,
    _bounds,
    generate_dataset,
    sample_permittivity,
)
from fdtdmesh.data.generate_v1 import make_scene as legacy_scene
from fdtdmesh.data.schema import validate_splits
from fdtdmesh.evaluation.metrics import spectrum
from fdtdmesh.evaluation.pipeline import (
    EvaluationConfig,
    converge_reference,
    grids,
    tail_diagnostic,
)


def test_broad_generator_is_seeded_multiscale_and_per_object():
    a, b = generate_dataset(16, 2026), generate_dataset(16, 2026)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    validate_splits(a)
    regular = [s for s in a if s.split in ("train", "validation", "test_iid")]
    counts = [len(s.geometry) for s in regular]
    assert min(counts) == 1 and max(counts) == 8
    aspect = [s.domain[1] / s.domain[0] for s in regular]
    assert min(aspect) < 0.5 and max(aspect) > 2.5
    eps = [m["epsilon_r"] for s in regular for m in s.materials]
    sigma = [m["sigma_e"] for s in regular for m in s.materials]
    assert min(eps) < 2 and max(eps) > 20
    assert 0.75 < np.mean(np.array(eps) <= 10) < 0.95
    assert 0 in sigma and max(sigma) > 1 and min(v for v in sigma if v) > 0
    spans = []
    kinds = set()
    for s in a:
        assert s.raster_shape == [128, 128]
        dielectric_names = [g["material"] for g in s.geometry if g.get("material", "PEC") != "PEC"]
        assert len(dielectric_names) == len(set(dielectric_names)) == len(s.materials)
        if s.split == "test_compositional":
            assert 9 <= len(s.geometry) <= 12
        elif s.split == "test_material_ood":
            assert all(m["epsilon_r"] > 30 and m["sigma_e"] > 10 for m in s.materials)
            assert len(s.materials) == len(s.geometry)
        elif s.split == "test_geometry_ood":
            assert all(g["kind"] == "polygon" and 5 <= len(g["vertices"]) <= 9 for g in s.geometry)
        for g in s.geometry:
            kinds.add(g["kind"])
            if g["kind"] != "pec_line":
                lo, hi = _bounds(g, np.array(s.domain))
                spans.extend(hi - lo)
        sim = s.build(s.budgets[0])
        for p in s.sources + s.receivers:
            eps, _, _, pec = sim.sample([p["x"]], [p["y"]])
            assert eps.item() == 1 and not pec.item()
    assert max(spans) / min(spans) > 8
    assert kinds == {"rectangle", "circle", "triangle", "polygon", "pec_line"}
    # Each receiver index, as well as the source, must vary across the whole interior.
    probes = np.array(
        [
            [[p["x"] / s.domain[0], p["y"] / s.domain[1]] for p in s.sources + s.receivers]
            for s in regular
        ]
    )
    assert np.all((probes >= 0.19) & (probes <= 0.81))
    assert np.all(np.ptp(probes, axis=0) > 0.5)
    for index in range(4):
        assert len(np.unique(probes[:, index, :] > 0.5, axis=0)) == 4
    for scene_probes in probes:
        distance = np.linalg.norm(scene_probes[:, None] - scene_probes[None, :], axis=-1)
        assert np.all(distance[np.triu_indices(4, k=1)] >= 0.08)


def test_permittivity_mixture_and_held_out_tail():
    cfg = GenerationConfig()
    rng = np.random.default_rng(71)
    draws = np.array([sample_permittivity(rng, cfg, "train") for _ in range(10000)])
    assert 0.83 < np.mean(draws <= 10) < 0.87
    assert draws.min() >= 1 and draws.max() <= 30
    assert draws.min() < 1.01 and draws.max() > 29
    # Log-uniform within each component, rather than just correct mixture counts.
    assert abs(np.mean(np.log(draws[draws <= 10])) - np.log(10) / 2) < 0.03
    assert abs(np.mean(np.log(draws[draws > 10])) - np.log(300) / 2) < 0.03
    for probability, bounds in ((0, (10, 30)), (1, (1, 10))):
        custom = replace(cfg, epsilon_core_probability=probability)
        values = [sample_permittivity(rng, custom, "train") for _ in range(100)]
        assert min(values) >= bounds[0] and max(values) <= bounds[1]
    ood = [sample_permittivity(rng, cfg, "test_material_ood") for _ in range(100)]
    assert min(ood) >= 36 and max(ood) <= 120


@pytest.mark.parametrize(
    "options",
    [
        {"epsilon_core_max": 0.5},
        {"epsilon_core_max": 31},
        {"epsilon_core_max": float("nan")},
        {"epsilon_core_probability": -0.1},
        {"epsilon_core_probability": 1.1},
        {"epsilon_core_probability": float("nan")},
    ],
)
def test_invalid_permittivity_mixture(options):
    with pytest.raises(ValueError, match="permittivity mixture"):
        GenerationConfig(**options)


def test_generator_configuration_limits_and_custom_counts():
    with pytest.raises(ValueError, match="four raster pixels"):
        GenerationConfig(raster_size=64)
    with pytest.raises(ValueError):
        GenerationConfig(min_objects=4, max_objects=2)
    scenes = generate_dataset(1, 15, config=GenerationConfig(min_objects=2, max_objects=2))
    assert all(len(s.geometry) == 2 for s in scenes if s.split != "test_compositional")


def test_ringdown_extends_duration_and_restarts_spatial_checks():
    spec = legacy_scene(3, "train", 1)
    base = spec.t_end
    calls = []
    config = EvaluationConfig(reference_levels=(8, 16, 32), samples=257)

    def runner(scene, budget, config, **kw):
        calls.append((scene.t_end, budget[0]))
        t = grids(scene, config)[0][1:]
        values = np.exp(-t / (base * 0.25)) * np.sin(2 * np.pi * 8 * t / base)
        return SimpleNamespace(
            times=t, receivers={0: values[:, None] * (1 + 1 / budget[0] ** 2)}, diagnostics={}
        )

    status, latest = converge_reference(spec, config, runner=runner)
    assert status["status"] == "converged" and status["duration"] == 2 * base
    assert calls == [(base, 8), (2 * base, 8), (2 * base, 16), (2 * base, 32)]
    assert status["duration_history"][0]["status"] == "time_unsettled"
    assert len(latest[1]) == len(grids(replace(spec, t_end=status["duration"]), config)[0])


def test_persistent_resonance_remains_unsettled_with_sampling_growth():
    spec = legacy_scene(3, "train", 1)
    config = EvaluationConfig(reference_levels=(8, 16, 32), samples=33, max_duration_extensions=1)

    def runner(scene, budget, config, **kw):
        t = grids(scene, config)[0][1:]
        return SimpleNamespace(
            times=t,
            receivers={0: np.sin(2 * np.pi * scene.f_max * 0.4 * t)[:, None]},
            diagnostics={},
        )

    status, _ = converge_reference(spec, config, runner=runner)
    assert status["status"] == "time_unsettled" and status["accepted_budget"] is None
    assert status["duration"] == 2 * spec.t_end
    a, b = grids(spec, config)
    assert len(a) > config.samples and len(b) > config.frequency_count
    with pytest.raises(ValueError, match="resource limit"):
        grids(spec, EvaluationConfig(max_observation_samples=10))


def test_tail_waits_for_excitation_and_chunked_dft_matches_direct():
    spec = legacy_scene(1, "train", 1)
    spec.t_end = 2 / spec.f_max
    assert not tail_diagnostic(spec, np.zeros((100, 3)), EvaluationConfig())["settled"]
    t = np.linspace(0, 1, 501)
    values = np.c_[np.sin(2 * np.pi * 4 * t), np.cos(2 * np.pi * 9 * t)]
    f = np.linspace(0, 20, 130)
    expected = (np.exp(-2j * np.pi * f[:, None] * t) * np.hanning(len(t)) * (t[1] - t[0])) @ values
    np.testing.assert_allclose(spectrum(values, t, f), expected, atol=1e-14)


def test_extended_reference_and_candidates_share_window(tmp_path, monkeypatch):
    from fdtdmesh import Mesh
    from fdtdmesh.data.schema import write_manifest
    from fdtdmesh.evaluation import pipeline

    spec = legacy_scene(3, "test_iid", 1)
    spec.budgets = [[64, 64]]
    manifest = tmp_path / "input.json"
    write_manifest(manifest, [spec], generation={"version": 1})
    base = spec.t_end
    candidate_times = []

    def run(scene, budget, config, **kw):
        if not kw.get("reference"):
            candidate_times.append(scene.t_end)
        t = grids(scene, config)[0][1:]
        values = np.exp(-t / (base * 0.25)) * np.sin(2 * np.pi * 8 * t / base)
        return SimpleNamespace(
            times=t,
            receivers={0: values[:, None]},
            diagnostics={"cell_updates": 1, "gpu_ms": 1},
            mesh=Mesh(np.linspace(0, scene.domain[0], 3), np.linspace(0, scene.domain[1], 3)),
            receiver_coordinates={0: np.array([[spec.receivers[0]["x"], spec.receivers[0]["y"]]])},
        )

    original = pipeline.converge_reference
    monkeypatch.setattr(pipeline, "converge_reference", lambda s, c: original(s, c, runner=run))
    monkeypatch.setattr(pipeline, "run_scene", run)
    report = pipeline.evaluate_dataset(
        manifest, tmp_path / "out", config=EvaluationConfig(reference_levels=(8, 16, 32))
    )
    assert report["scenes"][0]["duration"] == 2 * base
    assert candidate_times == [2 * base, 2 * base]
    for name in ("reference_latest.npz", "uniform_64_64.npz", "heuristic_64_64.npz"):
        with np.load(tmp_path / "out" / spec.scene_id / name) as arrays:
            assert arrays["times"][-1] == 2 * base


@pytest.mark.cuda
@pytest.mark.parametrize("strategy", ["uniform", "heuristic", "cnn"])
def test_broad_scene_runs_each_cuda_mesh_strategy(tmp_path, strategy):
    import torch

    from fdtdmesh.data.generate import make_scene
    from fdtdmesh.evaluation.pipeline import run_scene
    from fdtdmesh.ml import ResUNet, save_model
    from fdtdmesh.solver.tmz import cuda_available

    if not cuda_available():
        pytest.skip("CUDA unavailable")
    scene = make_scene(42, "test_iid")
    path = tmp_path / "demo.pt"
    torch.manual_seed(42)
    save_model(path, ResUNet(4), raster_shape=(128, 128))
    result = run_scene(scene, [64, 64], EvaluationConfig(), strategy=strategy, checkpoint=path)
    assert result.diagnostics["stepping_transfers"] == 0
    assert result.times[-1] >= scene.t_end
    assert all(np.isfinite(v).all() for v in result.fields.values())
    assert max(np.max(abs(v)) for v in result.receivers.values()) > 1e-6
    assert max(result.diagnostics[axis + "_grading"] for axis in ("x", "y")) <= 1.4 + 1e-8
