"""Launch/resume the mixed-budget CNN physics pilot; --status reads live results."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/physics_pilot_mixed_v7"
BUDGETS = ((64, 64), (128, 128), (80, 80), (112, 112), (72, 104), (104, 72))
FAMILIES = (
    "separated", "single_dielectric", "contact", "dielectric_gap",
    "overlap", "single_pec", "nested", "pec_dielectric",
)
STRATEGIES = ("uniform", "quasi_uniform", "heuristic", "cnn")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_scenes(scenes):
    selected = []
    for family in FAMILIES:
        candidates = sorted(
            (s for s in scenes if s.split == "validation" and s.family == family),
            key=lambda s: s.scene_id,
        )
        if not candidates:
            raise ValueError(f"Missing validation family {family}")
        selected.append(candidates[0])
    return selected


def prepare(output):
    from fdtdmesh.data.schema import read_manifest
    from fdtdmesh.mesh import MESH_POLICY
    from fdtdmesh.physics import _read_reference
    from fdtdmesh.uniform import BASELINE_POLICY

    training = ROOT / "artifacts/training_mixed_v7_3000"
    checkpoint = training / "pretraining/best.pt"
    manifest_path = ROOT / "artifacts/reference_combined_v7_3000/accepted_manifest.json"
    manifest, scenes = read_manifest(manifest_path)
    audit = read(training / "reference_audit.json")
    if audit["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Reference audit dataset mismatch")
    entries = []
    for scene in select_scenes(scenes):
        reference = Path(audit["references"][scene.scene_id])
        status, *_ = _read_reference(scene, reference, arrays_name="reference_latest.npz")
        if status["status"] != "converged":
            raise ValueError(f"Reference is not converged: {scene.scene_id}")
        entries.append(dict(
            scene=scene.to_dict(), reference=str(reference),
            config=read(reference.parent / "run.json")["config"],
            reference_sha256={name: digest(reference / name) for name in (
                "reference.json", "reference_latest.npz", "evaluated_scene.json"
            )},
        ))
    sources = sorted((ROOT / "src/fdtdmesh").rglob("*.py"))
    sources += sorted((ROOT / "src/fdtdmesh").rglob("*.so"))
    sources.append(Path(__file__))
    plan = dict(
        version=1, dataset_id=manifest["dataset_id"], checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), budgets=[list(b) for b in BUDGETS],
        strategies=list(STRATEGIES), scenes=entries, max_cell_updates=256_000_000_000,
        mesh_policy=MESH_POLICY, baseline_policy=BASELINE_POLICY,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Pilot inputs changed; use a new output directory")
    write(path, plan)
    return plan


def run_worker(output, index):
    import numpy as np
    import torch
    from dataclasses import replace
    from fdtdmesh.data.schema import SceneSpec
    from fdtdmesh.evaluation.metrics import sample_observables
    from fdtdmesh.evaluation.pipeline import EvaluationConfig, metrics, run_scene, tail_diagnostic
    from fdtdmesh.ml import load_model
    from fdtdmesh.physics import _predict_density, _read_reference

    plan = read(output / "plan.json")
    entry = plan["scenes"][index]
    spec = SceneSpec.from_dict(entry["scene"])
    directory = output / spec.scene_id
    directory.mkdir(exist_ok=True)
    status, effective, times, frequencies, reference = _read_reference(
        spec, Path(entry["reference"]), arrays_name="reference_latest.npz"
    )
    if status["status"] != "converged":
        raise ValueError("Pilot requires a converged reference")
    config = replace(EvaluationConfig(**entry["config"]), max_cell_updates=plan["max_cell_updates"])
    device = torch.device("cuda:0")
    model, metadata = load_model(plan["checkpoint"], device=device)
    for budget in plan["budgets"]:
        for strategy in plan["strategies"]:
            key = f"{budget[0]}x{budget[1]}-{strategy}"
            path = directory / f"{key}.json"
            if path.exists():
                continue
            started = time.time()
            write(directory / "active.json", dict(case=key, started_unix=started))
            row = dict(scene_id=spec.scene_id, scene_hash=spec.content_hash,
                       family=spec.family, split=spec.split, budget=budget, strategy=strategy,
                       budget_group="seen" if tuple(budget) in BUDGETS[:2] else "heldout",
                       reference_budget=status["accepted_budget"], duration=effective.t_end,
                       status="ok")
            try:
                kwargs = dict(strategy=strategy)
                if strategy == "cnn":
                    kwargs = dict(strategy="density", density=_predict_density(
                        effective, budget, model, metadata, device
                    ))
                result = run_scene(effective, budget, config, **kwargs)
                observations = sample_observables(result, times)
                row.update(metrics=metrics(observations, reference, times, frequencies, config),
                           diagnostics=result.diagnostics,
                           tail=tail_diagnostic(effective, observations, config))
                np.savez_compressed(directory / f"{key}.npz", times=times,
                                    waveforms=observations, x=result.mesh.x, y=result.mesh.y)
            except (ValueError, RuntimeError) as error:
                row.update(status="failed", error=str(error), error_type=type(error).__name__)
            row["wall_seconds"] = time.time() - started
            write(path, row)
            print(json.dumps(row), flush=True)
    write(directory / "active.json", dict(status="complete"))


def collect(output):
    plan = read(output / "plan.json")
    rows = []
    for entry in plan["scenes"]:
        directory = output / entry["scene"]["scene_id"]
        for budget in plan["budgets"]:
            for strategy in plan["strategies"]:
                path = directory / f"{budget[0]}x{budget[1]}-{strategy}.json"
                if path.exists():
                    rows.append(read(path))
    return dict(planned=len(plan["scenes"]) * len(plan["budgets"]) * len(plan["strategies"]),
                completed=len(rows), successful=sum(r["status"] == "ok" for r in rows),
                failed=sum(r["status"] != "ok" for r in rows), rows=rows)


def coordinate(output, gpus):
    # A second accidental launch cannot start duplicate solver workers.
    with (output / "pilot.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(output / "plan.json")
        pending = list(range(len(plan["scenes"])))
        active, exits = {}, {}
        started = time.time()
        while pending or active:
            for gpu, (index, process) in list(active.items()):
                code = process.poll()
                if code is not None:
                    exits[index] = code
                    del active[gpu]
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                index = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
                with (output / f"worker{index}.log").open("a") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(Path(__file__)), "--output", str(output),
                         "--worker", str(index)], cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    )
                active[gpu] = (index, process)
            report = collect(output)
            state = "running" if pending or active else (
                "failed" if any(exits.values()) or report["completed"] != report["planned"]
                else "complete"
            )
            report.update(state=state, started_unix=started, updated_unix=time.time(),
                          worker_exit_codes=exits,
                          active={gpu: dict(scene_index=i, pid=p.pid) for gpu, (i, p) in active.items()})
            write(output / "report.json", report)
            if pending or active:
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--coordinate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.status:
        report = read(output / "report.json")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    elif args.worker is not None:
        run_worker(output, args.worker)
    elif args.coordinate:
        coordinate(output, args.gpus)
    else:
        if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
            raise ValueError("GPU indices must be distinct and nonnegative")
        print("Validating pilot scenes and reference identities...", flush=True)
        prepare(output)
        if args.prepare_only:
            print(f"Pilot prepared: {output}")
            return
        with (output / "coordinator.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--coordinate", "--output", str(output),
                 "--gpus", *map(str, args.gpus)], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write(output / "launch.json", dict(pid=process.pid, started_unix=time.time(), gpus=args.gpus))
        print(f"Physics pilot launched: PID {process.pid}; {output}")


if __name__ == "__main__":
    main()
