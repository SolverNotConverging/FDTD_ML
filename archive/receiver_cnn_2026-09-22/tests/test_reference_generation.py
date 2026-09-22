import json
from types import SimpleNamespace

import numpy as np
import pytest

from fdtdmesh.data.generate import generate_splits
from fdtdmesh.data.schema import write_manifest
from fdtdmesh.evaluation import EvaluationConfig, generate_references


def test_generate_exact_reference_split_counts_and_is_reproducible():
    counts = {"train": 3, "validation": 2, "test_iid": 4}
    first = generate_splits(counts, seed=91)
    second = generate_splits(counts, seed=91)
    assert [scene.to_dict() for scene in first] == [scene.to_dict() for scene in second]
    assert {split: sum(scene.split == split for scene in first) for split in counts} == counts
    assert {scene.split for scene in first} == set(counts)
    assert [scene.scene_id for scene in first if scene.split == "validation"] == [
        "validation-00000",
        "validation-00001",
    ]


@pytest.mark.parametrize(
    "counts",
    [{}, {"not_a_split": 1}, {"train": 0}, {"train": True}],
)
def test_generate_split_counts_reject_invalid_contract(counts):
    with pytest.raises(ValueError):
        generate_splits(counts)


def test_reference_generation_resumes_committed_scenes(tmp_path, monkeypatch):
    scenes = generate_splits({"train": 1, "validation": 1, "test_iid": 1}, seed=13)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, scenes, generation={"seed": 13})
    calls = []

    def fake_convergence(spec, config):
        calls.append(spec.scene_id)
        evaluated = SimpleNamespace(t_end=spec.t_end * 2, f_min=spec.f_min, f_max=spec.f_max)
        from fdtdmesh.evaluation.pipeline import grids

        times, _ = grids(evaluated, config)
        result = SimpleNamespace(
            mesh=SimpleNamespace(
                x=np.linspace(0, spec.domain[0], 129), y=np.linspace(0, spec.domain[1], 129)
            ),
            receiver_coordinates={
                i: np.array([[receiver["x"], receiver["y"]]])
                for i, receiver in enumerate(spec.receivers)
            },
        )
        return (
            {
                "status": "converged",
                "levels": [],
                "accepted_budget": [128, 128],
                "duration": spec.t_end * 2,
                "duration_history": [],
            },
            (result, np.zeros((len(times), len(spec.receivers)))),
        )

    from fdtdmesh.evaluation import references

    monkeypatch.setattr(references, "converge_reference", fake_convergence)
    output = tmp_path / "references"
    report = generate_references(manifest, output)
    assert calls == ["train-00000", "validation-00000", "test_iid-00000"]
    assert report["coverage"]["accepted"] == 3
    assert report["selected_scene_count"] == 3
    assert (output / "summary.md").is_file()
    for scene in scenes:
        status = json.loads((output / scene.scene_id / "reference.json").read_text())
        evaluated = json.loads((output / scene.scene_id / "evaluated_scene.json").read_text())
        assert status["scene_hash"] == scene.content_hash
        assert evaluated["t_end"] == pytest.approx(scene.t_end * 2)

    calls.clear()
    resumed = generate_references(manifest, output)
    assert calls == []
    assert resumed["coverage"] == report["coverage"]


def test_reference_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    scenes = generate_splits({"train": 1}, seed=27)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, scenes, generation={"seed": 27})

    from fdtdmesh.evaluation import references

    monkeypatch.setattr(
        references,
        "converge_reference",
        lambda spec, config: (
            {
                "status": "failed",
                "levels": [],
                "accepted_budget": None,
                "duration": spec.t_end,
                "duration_history": [],
            },
            None,
        ),
    )
    output = tmp_path / "references"
    generate_references(manifest, output, splits=("train",))
    changed = EvaluationConfig(relative_tolerance=0.01)
    with pytest.raises(ValueError, match="different dataset or configuration"):
        generate_references(manifest, output, config=changed, splits=("train",))


def test_reference_generation_reuses_only_exact_converged_scene(tmp_path):
    scenes = generate_splits({"train": 2}, seed=39)
    manifest = tmp_path / "expanded.json"
    value = write_manifest(manifest, scenes, generation={"seed": 39})
    reuse = tmp_path / "reuse"
    source = reuse / scenes[0].scene_id
    source.mkdir(parents=True)
    (reuse / "run.json").write_text(json.dumps({"dataset_id": "older-dataset"}), encoding="utf-8")
    status = {
        "status": "converged",
        "accepted_budget": [128, 128],
        "duration": scenes[0].t_end,
        "duration_history": [],
        "levels": [],
        "scene_id": scenes[0].scene_id,
        "scene_hash": scenes[0].content_hash,
        "split": "train",
        "family": scenes[0].family,
        "wall_seconds": 1.0,
    }
    (source / "scene.json").write_text(json.dumps(scenes[0].to_dict()), encoding="utf-8")
    (source / "evaluated_scene.json").write_text(json.dumps(scenes[0].to_dict()), encoding="utf-8")
    (source / "reference.json").write_text(json.dumps(status), encoding="utf-8")
    np.savez_compressed(source / "reference_latest.npz", waveforms=np.zeros((2, 1)))

    def failed_runner(*args, **kwargs):
        raise RuntimeError("deliberate test failure")

    output = tmp_path / "output"
    report = generate_references(
        manifest,
        output,
        splits=("train",),
        reuse_roots=(reuse,),
        runner=failed_runner,
    )
    assert report["dataset_id"] == value["dataset_id"]
    assert report["scenes"][0]["status"] == "converged"
    assert report["scenes"][0]["reused"] is True
    assert report["scenes"][0]["reused_from_dataset_id"] == "older-dataset"
    assert report["scenes"][1]["status"] == "failed"
