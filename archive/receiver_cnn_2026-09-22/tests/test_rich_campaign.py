import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data import campaign
from fdtdmesh.data.generate_v1 import make_scene
from fdtdmesh.data.generate_v5 import RichConfig, feature_audit, make_rich_scene
from fdtdmesh.evaluation import EvaluationConfig, converge_reference


def test_campaign_reference_level_ceiling():
    assert campaign.reference_config(2048).reference_levels == (128, 256, 512, 1024, 2048)
    assert not campaign.reference_config(2048, extend_nonconverged=False).extend_nonconverged
    with pytest.raises(ValueError, match="Maximum reference level"):
        campaign.reference_config(512)


def test_rich_scenes_are_resolved_nonmagnetic_and_reproducible():
    scenes = [make_rich_scene(2026, "train", i) for i in range(4)]
    assert scenes[2].content_hash == make_rich_scene(2026, "train", 2).content_hash
    assert {s.family for s in scenes} == {"separated", "contact", "overlap", "nested"}
    materials = [m for s in scenes for m in s.materials]
    assert max(m["epsilon_r"] for m in materials) > 20
    assert all(m["mu_r"] == 1 and m["sigma_h"] == 0 for m in materials)
    assert max(m["sigma_e"] for m in materials) > 0
    for s in scenes:
        assert 8 <= sum(g["kind"] != "pec_line" for g in s.geometry) <= 16
        assert any(g.get("material") == "PEC" for g in s.geometry)
        audit = feature_audit(s)
        assert audit["minimum_core_pixels"] >= 3
        assert min(min(o["cells_at_1024"]) for o in audit["objects"]) >= 128 - 1e-8
        assert s.anchor_probes
        for level in (128, 256):
            simulation = s.build([level, level], reference=True)
            mesh = simulation.mesh_uniform()
            for probe in [*s.sources, *s.receivers]:
                assert probe["x"] in simulation.x_anchors and probe["y"] in simulation.y_anchors
                assert probe["x"] in mesh.x and probe["y"] in mesh.y
    assert make_rich_scene(2026, "validation", 0).seed != scenes[0].seed


def test_composite_audit_rejects_tiny_or_hidden_objects():
    scene = make_rich_scene(2026, "train", 0)
    tiny = dict(
        kind="circle",
        material="PEC",
        center=[0.5 * d for d in scene.domain],
        radius=min(scene.domain) / 1000,
    )
    with pytest.raises(ValueError):
        feature_audit(replace(scene, geometry=[*scene.geometry, tiny]))
    with pytest.raises(ValueError):
        feature_audit(replace(scene, geometry=[*scene.geometry, scene.geometry[0]]))


def test_pec_lines_and_probes_survive_uniform_reference_refinement():
    config = replace(RichConfig(), pec_line_probability=1)
    scene = make_rich_scene(2026, "train", 0, config=config)
    lines = [g for g in scene.geometry if g["kind"] == "pec_line"]
    assert lines
    for n in (128, 256):
        simulation = scene.build([n, n], reference=True)
        mesh = simulation.mesh_uniform()
        for line in lines:
            for axis in ("x", "y"):
                for coordinate in np.atleast_1d(line[axis]):
                    assert coordinate in getattr(simulation, f"{axis}_anchors")
                    assert coordinate in getattr(mesh, axis)
        *_, pec = simulation.sample(mesh.x, mesh.y)
        assert pec.any()
    legacy = make_scene(2026, "train", 1)
    assert "anchor_probes" not in legacy.to_dict()


def test_reference_minimum_and_spatial_exhaustion_extend_time():
    scene = make_scene(7, "train", 0)
    config = EvaluationConfig(
        reference_levels=(8, 16, 32, 64),
        minimum_reference_level=64,
        max_duration_extensions=1,
        extend_nonconverged=True,
        tail_relative_tolerance=None,
    )
    calls = []

    def run(spec, budget, config, **kwargs):
        calls.append((spec.t_end, budget[0]))
        times = np.linspace(0, spec.t_end, config.samples)[1:]
        # First duration is spatially nonconverged; second duration agrees.
        amplitude = budget[0] if spec.t_end == scene.t_end else 1
        return SimpleNamespace(
            times=times,
            receivers={0: amplitude * np.sin(2 * np.pi * times / spec.t_end)[:, None]},
            diagnostics={},
        )

    events = []
    status, _ = converge_reference(scene, config, runner=run, progress=events.append)
    assert status["status"] == "converged" and status["accepted_budget"] == [64, 64]
    assert len(status["duration_history"]) == 2 and len(calls) == 8
    assert events[-1]["event"] == "duration_completed"
    # Same spatially divergent data must stop at the explicit duration ceiling.
    limited = replace(config, max_duration_extensions=0)
    status, _ = converge_reference(scene, limited, runner=run)
    assert status["status"] == "nonconverged"


def test_campaign_replaces_rejects_until_quota_and_resumes(tmp_path, monkeypatch):
    config = dict(
        seed=2026,
        gpus=[0],
        targets=dict(train=2, validation=1, test_iid=1),
        generator=asdict(RichConfig()),
        reference=asdict(campaign.reference_config()),
        campaign_id="test",
        max_candidates_per_split_per_lane=20,
        scene_wall_seconds=30,
    )
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    calls = []

    def scene(seed, split, index, config):
        spec = make_scene(seed + index, split, index)
        spec.scene_id = f"{split}-v{campaign.GENERATOR_VERSION}-{index:06d}"
        spec.generation_attempts = 1
        return spec

    monkeypatch.setattr(campaign, "make_rich_scene", scene)
    monkeypatch.setattr(campaign, "feature_audit", lambda *args: {})

    class Process:
        def __init__(self, command, **kwargs):
            from pathlib import Path

            root = Path(command[-1])
            record = json.loads((root / "manifest.json").read_text())["scenes"][0]
            identity = record["scene_id"]
            calls.append(identity)
            path = root / "references" / identity
            path.mkdir(parents=True, exist_ok=True)
            state = (
                "nonconverged"
                if identity == f"train-v{campaign.GENERATOR_VERSION}-000000"
                else "converged"
            )
            (path / "reference.json").write_text(json.dumps(dict(status=state)))

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    campaign._worker(tmp_path, 0)
    summary = campaign.summarize(tmp_path)
    assert summary["accepted"] == 4 and summary["counts"]["nonconverged"] == 1
    assert f"train-v{campaign.GENERATOR_VERSION}-000004" in calls
    n = len(calls)
    campaign._worker(tmp_path, 0)
    assert len(calls) == n
    assert (tmp_path / "lane0/complete.json").exists()


def test_split_quota_totals_and_no_false_completion(tmp_path):
    for n in [1, 7, 128, 1024]:
        assert sum(sum(campaign.quotas(n, 4, lane)) for lane in range(4)) == n
    with pytest.raises(ValueError, match="quota unmet"):
        campaign.finalize(tmp_path, dict(targets={"train": 1024}, campaign_id="test", generator={}))


def test_material_resolution_blocks_coarse_high_index_and_loss():
    from fdtdmesh.constants import C0, EPS0, MU0
    from fdtdmesh.evaluation.pipeline import material_resolution_level

    scene = make_scene(4, "train", 0)
    scene.domain = [0.03, 0.03]
    scene.f_max = C0 / 0.03
    scene.f_min = scene.f_max / 20
    scene.materials = [dict(name="test", epsilon_r=30, mu_r=30, sigma_e=0, sigma_h=0)]
    config = EvaluationConfig(wavelength_cells=16, attenuation_cells=4)
    assert 480 <= material_resolution_level(scene, config) <= 481
    omega = 2 * np.pi * scene.f_max
    scene.materials[0].update(sigma_e=omega * EPS0 * 30, sigma_h=omega * MU0 * 30)
    assert material_resolution_level(scene, config) >= 753
