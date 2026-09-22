import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def pilot_module():
    path = Path(__file__).resolve().parents[1] / "scripts/start_physics_pilot_v7.py"
    spec = importlib.util.spec_from_file_location("physics_pilot_v7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_keeps_all_families_and_excludes_training_and_test():
    pilot = pilot_module()
    scenes = [SimpleNamespace(family=f, split=split, scene_id=f"{i}-{f}-{split}")
              for f in pilot.FAMILIES for split in ("train", "validation", "test_iid")
              for i in (1, 0)]
    selected = pilot.select_scenes(list(reversed(scenes)))
    assert [s.family for s in selected] == list(pilot.FAMILIES)
    assert all(s.split == "validation" and s.scene_id.startswith("0-") for s in selected)


def test_worker_reuses_reference_duration_records_failures_and_resumes(tmp_path, monkeypatch):
    import fdtdmesh.data.schema as schema
    import fdtdmesh.evaluation.metrics as metric_module
    import fdtdmesh.evaluation.pipeline as pipeline
    import fdtdmesh.ml as ml
    import fdtdmesh.physics as physics

    pilot = pilot_module()
    scene = SimpleNamespace(scene_id="validation-example", content_hash="identity",
                            split="validation", family="single_pec", t_end=1.0)
    effective = SimpleNamespace(t_end=3.0)
    monkeypatch.setattr(schema.SceneSpec, "from_dict", lambda _: scene)
    times = np.linspace(0, 3, 5)
    reference = np.zeros((5, 1))
    monkeypatch.setattr(physics, "_read_reference", lambda *a, **k: (
        dict(status="converged", accepted_budget=[1024, 1024]),
        effective, times, np.array([1.0]), reference,
    ))
    monkeypatch.setattr(ml, "load_model", lambda *a, **k: (None, {}))
    monkeypatch.setattr(physics, "_predict_density", lambda *a: (np.ones(2), np.ones(2)))
    calls = []

    def run(spec, budget, config, **kwargs):
        assert spec.t_end == 3.0
        assert config.material_averaging == "sampled"
        assert config.max_cell_updates == 123456
        calls.append((tuple(budget), kwargs["strategy"]))
        if len(calls) == 3:
            raise RuntimeError("Projection timed out")
        return SimpleNamespace(diagnostics=dict(cell_updates=500),
                               mesh=SimpleNamespace(x=np.arange(3), y=np.arange(3)))

    monkeypatch.setattr(pipeline, "run_scene", run)
    monkeypatch.setattr(metric_module, "sample_observables", lambda *a: reference)
    monkeypatch.setattr(pipeline, "metrics", lambda *a: dict(waveform_l2_max=0.1))
    monkeypatch.setattr(pipeline, "tail_diagnostic", lambda *a: dict(settled=True))
    pilot.write(tmp_path / "plan.json", dict(
        scenes=[dict(scene=dict(scene_id=scene.scene_id), reference="unused",
                     config=dict(material_averaging="sampled"))], checkpoint="unused",
        max_cell_updates=123456, budgets=[[64, 64], [72, 104]], strategies=list(pilot.STRATEGIES),
    ))
    pilot.run_worker(tmp_path, 0)
    report = pilot.collect(tmp_path)
    assert (report["completed"], report["successful"], report["failed"]) == (8, 7, 1)
    assert ((72, 104), "density") in calls
    assert next(r for r in report["rows"] if r["status"] == "failed")["error"] == "Projection timed out"
    pilot.run_worker(tmp_path, 0)
    assert len(calls) == 8
