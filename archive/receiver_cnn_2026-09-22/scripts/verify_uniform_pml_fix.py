"""Recheck failed intermediate-budget uniform baselines on CUDA before restart."""

import json
from pathlib import Path
import time

import numpy as np

from fdtdmesh.data.schema import SceneSpec, digest
from fdtdmesh.evaluation.metrics import sample_observables
from fdtdmesh.evaluation.pipeline import EvaluationConfig, metrics, run_scene
from fdtdmesh.physics import _read_reference

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT / "artifacts/physics_distillation_mixed_v7"
    output = ROOT / "artifacts/physics_uniform_pml_fix"
    workflow = json.loads((old / "workflow.json").read_text())
    manifest = json.loads(Path(workflow["manifest"]).read_text())
    assert digest(manifest["scenes"]) == workflow["dataset_id"]
    scenes = {s["scene_id"]: SceneSpec.from_dict(s) for s in manifest["scenes"]
              if s["scene_id"] in workflow["selection"]}
    cases = []
    unchanged = None
    for path in sorted(old.glob("lane*/search/*/*/result.json")):
        row = json.loads(path.read_text())
        uniform = next((r for r in row["candidates"] if r["name"] == "uniform"), {})
        if uniform.get("error") == "Uniform reference budget must align PML interfaces":
            cases.append((path, False))
        elif unchanged is None and "sample" in row:
            unchanged = path
    assert cases and unchanged is not None
    cases.append((unchanged, True))
    results = []
    config = EvaluationConfig(**workflow["evaluation"])
    for path, compare_saved in cases:
        sid = path.parent.parent.name
        budget = list(map(int, path.parent.name.split("_")))
        status, effective, times, frequencies, reference = _read_reference(
            scenes[sid], Path(workflow["references"]) / sid, arrays_name="reference_latest.npz"
        )
        assert status["status"] == "converged"
        started = time.time()
        simulation = run_scene(effective, budget, config, strategy="uniform")
        observations = sample_observables(simulation, times)
        assert np.isfinite(observations).all()
        assert simulation.diagnostics["backend"] == "cuda"
        for axis, count, length in zip("xy", budget, effective.domain):
            np.testing.assert_allclose(np.diff(getattr(simulation.mesh, axis)), length / count,
                                       rtol=1e-12, atol=0)
        if compare_saved:
            with np.load(path.parent / "uniform.npz") as cached:
                np.testing.assert_array_equal(simulation.mesh.x, cached["x"])
                np.testing.assert_array_equal(simulation.mesh.y, cached["y"])
                np.testing.assert_allclose(observations, cached["waveforms"], rtol=1e-6, atol=1e-12)
        row = dict(scene_id=sid, budget=budget, status="ok", checked_saved_waveforms=compare_saved,
                   wall_seconds=time.time() - started,
                   metrics=metrics(observations, reference, times, frequencies, config),
                   pml_snapping=simulation.diagnostics["pml_snapping"])
        results.append(row)
        output.mkdir(exist_ok=True)
        (output / "cuda_verification.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    print(f"Passed {len(cases)-1} previously failed cases and one unchanged-grid comparison")


if __name__ == "__main__":
    main()
