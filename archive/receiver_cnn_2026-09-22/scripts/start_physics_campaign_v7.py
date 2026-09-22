"""Mixed-budget physics search followed by CNN distillation on dense/sparse scenes."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/physics_distillation_mixed_v7_pml_v2"
TRAINING = ROOT / "artifacts/training_mixed_v7_3000"
MANIFEST = ROOT / "artifacts/reference_combined_v7_3000/accepted_manifest.json"
FAMILIES = ("separated", "contact", "overlap", "nested", "single_dielectric",
            "dielectric_gap", "single_pec", "pec_dielectric")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def select(scenes, excluded):
    selected, lanes = {}, [[] for _ in range(4)]
    for split, count in (("train", 16), ("validation", 4)):
        for family in FAMILIES:
            candidates = sorted((s for s in scenes if s.split == split and s.family == family
                                 and s.scene_id not in excluded), key=lambda s: s.scene_id)
            if len(candidates) < count:
                raise ValueError(f"Not enough {split}/{family} scenes")
            for i, scene in enumerate(candidates[:count]):
                selected[scene.scene_id] = dict(split=split, family=family)
                lanes[i % 4].append(scene.scene_id)
    return selected, lanes


def prepare(output, reuse_from=None):
    from dataclasses import asdict, replace
    from fdtdmesh.budget_schedule import HELDOUT_BUDGETS
    from fdtdmesh.data.schema import provenance, read_manifest
    from fdtdmesh.evaluation.pipeline import EvaluationConfig
    from fdtdmesh.ml import load_model
    from fdtdmesh.physics import SearchConfig, _read_reference, _search_budget_entries, _sha256
    from fdtdmesh.training import TrainingConfig
    from fdtdmesh.uniform import BASELINE_POLICY
    from start_physics_distillation_v6 import _prepare_references

    manifest, scenes = read_manifest(MANIFEST)
    pilot = read(ROOT / "artifacts/physics_pilot_mixed_v7/plan.json")
    excluded = {entry["scene"]["scene_id"] for entry in pilot["scenes"]}
    selected, lanes = select(scenes, excluded)
    chosen = [s for s in scenes if s.scene_id in selected]
    entries = _search_budget_entries(chosen, budget_plan=read(TRAINING / "budget_plan.json"))
    for scene in chosen:
        pairs = [tuple(e["budget"]) for e in entries[scene.scene_id]]
        if any(not 48 <= n <= 128 for pair in pairs for n in pair):
            raise ValueError("Budgets must stay within 48–128")
        if scene.split == "train" and set(pairs) & set(HELDOUT_BUDGETS):
            raise ValueError("Reserved validation budgets leaked into training")
    checkpoint = TRAINING / "pretraining/best.pt"
    _, metadata = load_model(checkpoint)
    if metadata["dataset_version"] != manifest["dataset_id"]:
        raise ValueError("Checkpoint belongs to another dataset")
    reference_root = _prepare_references(
        output, manifest["dataset_id"], list(selected), TRAINING / "reference_audit.json"
    )
    evaluation = EvaluationConfig(**read(ROOT / "artifacts/reference_sparse_v7_1000/campaign.json")["reference"])
    evaluation = replace(evaluation, max_cell_updates=256_000_000_000)
    bindings = {}
    numerical_keys = ("material_averaging", "averaging_samples", "averaging_max_samples",
                      "averaging_tolerance", "amplitude_floor", "phase_gate",
                      "tail_relative_tolerance", "tail_fraction")
    for scene in chosen:
        directory = reference_root / scene.scene_id
        status, *_ = _read_reference(scene, directory, arrays_name="reference_latest.npz")
        if status["status"] != "converged":
            raise ValueError(f"Reference is not converged: {scene.scene_id}")
        source_config = read(directory.resolve().parent / "run.json")["config"]
        if any(source_config[k] != getattr(evaluation, k) for k in numerical_keys):
            raise ValueError(f"Reference numerical settings differ: {scene.scene_id}")
        bindings[scene.scene_id] = {name: _sha256(directory / name) for name in (
            "reference.json", "reference_latest.npz", "evaluated_scene.json"
        )}
    config = TrainingConfig(epochs=20, batch_size=4, learning_rate=3e-4, width=16,
                            repair_weight=0.0, projection_samples=8, seed=2026,
                            early_stopping_patience=8, sparse_sample_weight=2.0)
    teacher_path = TRAINING / "teacher/targets.npz"
    feasible = {(s["scene_id"], tuple(s["budget"]))
                for s in read(teacher_path.with_suffix(".json"))["samples"]}
    requested = sum(map(len, entries.values()))
    available = sum((sid, tuple(e["budget"])) in feasible for sid, es in entries.items() for e in es)
    workflow = dict(
        version=1, manifest=str(MANIFEST), dataset_id=manifest["dataset_id"],
        baseline_policy=BASELINE_POLICY,
        reuse_from=None if reuse_from is None else str(reuse_from),
        checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
        teacher_targets=str(teacher_path), teacher_targets_sha256=_sha256(teacher_path),
        selection=selected, lanes=lanes, budget_plan=dict(assignments=entries),
        references=str(reference_root), reference_bindings=bindings,
        evaluation=asdict(evaluation), search=asdict(SearchConfig()),
        distillation=asdict(config), requested_pairs=requested, feasible_teacher_pairs=available,
        source_sha256=provenance()["source_sha256"], launcher_sha256=_sha256(__file__),
        helper_sha256=_sha256(ROOT / "scripts/start_physics_distillation_v6.py"),
    )
    workflow = json.loads(json.dumps(workflow))
    path = output / "workflow.json"
    if path.exists() and read(path) != workflow:
        raise ValueError("Campaign identity changed; choose another output")
    write(path, workflow)
    return workflow


def reuse_unchanged_pairs(output, workflow, source):
    """Import only successful v1 pairs whose uniform grid/PML did not change."""
    import numpy as np
    from fdtdmesh.data.schema import SceneSpec
    from fdtdmesh.uniform import uniform_scene

    previous = read(source / "workflow.json")
    for key in ("dataset_id", "checkpoint_sha256", "teacher_targets_sha256", "selection",
                "lanes", "budget_plan", "reference_bindings", "evaluation", "search", "distillation"):
        if previous[key] != workflow[key]:
            raise ValueError(f"Cannot reuse pairs with a changed {key}")
    reused = []
    for lane in range(4):
        source_root = source / f"lane{lane}/search"
        if read(source_root / "run.json")["baseline_policy"] != "uniform_snapped_pec_and_quasi_uniform_v1":
            raise ValueError("Migration expects the original uniform/PML policy")
        for path in sorted(source_root.glob("*/*/result.json")):
            row = read(path)
            if "sample" not in row:
                continue  # Failed and unfinished pairs must be evaluated under v2.
            sample = row["sample"]
            reference = Path(workflow["references"]) / sample["scene_id"]
            effective = SceneSpec.from_dict(read(reference / "evaluated_scene.json"))
            scene = uniform_scene(effective, sample["budget"])
            if any(v["interface_shift"] != 0 for v in scene.mesh.metadata["pml_snapping"].values()):
                continue
            with np.load(path.parent / "uniform.npz") as arrays:
                if not (np.array_equal(scene.mesh.x, arrays["x"]) and np.array_equal(scene.mesh.y, arrays["y"])):
                    continue
            row["cache_origin"] = dict(directory=str(path.parent),
                baseline_policy="uniform_snapped_pec_and_quasi_uniform_v1",
                validation="Uniform grid identical; physical PML thickness unchanged",
                source_sha256=previous["source_sha256"])
            destination = output / f"lane{lane}/search" / path.parent.relative_to(source_root)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".importing")
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(path.parent, temporary)
                write(temporary / "result.json", row)
                temporary.rename(destination)
            reused.append(dict(scene_id=sample["scene_id"], budget=sample["budget"]))
    write(output / "reuse.json", dict(source=str(source), pairs=reused, count=len(reused)))


def progress(output):
    workflow = read(output / "workflow.json")
    completed = targets = failures = candidates = candidate_failures = 0
    for lane, scene_ids in enumerate(workflow["lanes"]):
        for sid in scene_ids:
            for entry in workflow["budget_plan"]["assignments"][sid]:
                nx, ny = entry["budget"]
                path = output / f"lane{lane}/search/{sid}/{nx}_{ny}/result.json"
                if not path.exists():
                    continue
                row = read(path)
                completed += 1
                targets += "sample" in row
                failures += row.get("status") == "failed"
                candidates += len(row["candidates"])
                candidate_failures += sum(r["status"] != "ok" for r in row["candidates"])
    return dict(requested_pairs=workflow["requested_pairs"], completed_pairs=completed,
                targets=targets, failed_pairs=failures, recorded_candidates=candidates,
                candidate_failures=candidate_failures, updated_unix=time.time())


def worker(output, lane):
    from fdtdmesh.evaluation.pipeline import EvaluationConfig
    from fdtdmesh.physics import SearchConfig, search_physics_targets

    workflow = read(output / "workflow.json")
    search_physics_targets(
        workflow["manifest"], workflow["teacher_targets"], workflow["checkpoint"],
        output / f"lane{lane}/search", references=workflow["references"],
        scene_ids=workflow["lanes"][lane], budget_plan=workflow["budget_plan"],
        evaluation_config=EvaluationConfig(**workflow["evaluation"]),
        search_config=SearchConfig(**workflow["search"]), device="cuda:0",
    )


def run(output, reuse_from=None):
    from start_physics_distillation_v6 import _merge_targets

    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / "stage.json", dict(stage="preparing", time=time.time()))
        workflow = prepare(output, reuse_from)
        if reuse_from is not None:
            reuse_unchanged_pairs(output, workflow, reuse_from)
        write(output / "stage.json", dict(stage="physics_search", time=time.time()))
        processes = []
        for lane in range(4):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with (output / f"lane{lane}.log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--output", str(output), "--worker", str(lane)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            processes.append(process)
        write(output / "workers.json", [dict(lane=i, pid=p.pid) for i, p in enumerate(processes)])
        while any(p.poll() is None for p in processes):
            write(output / "progress.json", progress(output))
            time.sleep(10)
        write(output / "progress.json", progress(output))
        codes = [p.returncode for p in processes]
        if any(codes):
            raise RuntimeError(f"Search worker failures: {codes}")
        write(output / "stage.json", dict(stage="merge_targets", time=time.time()))
        targets, metadata = _merge_targets(
            dict(dataset_id=workflow["dataset_id"]),
            [output / f"lane{i}/search" for i in range(4)], output,
        )
        counts = Counter((s["split"], workflow["selection"][s["scene_id"]]["family"])
                         for s in metadata["samples"])
        if any(counts[(split, family)] == 0 for split in ("train", "validation") for family in FAMILIES):
            raise ValueError("Missing target coverage for a split/family; inspect search outcomes")
        # Every requested pair must have a durable target or explicit failure outcome.
        report = progress(output)
        if report["completed_pairs"] != report["requested_pairs"]:
            raise ValueError("Search left unaccounted scene/budget pairs")
        write(output / "coverage.json", dict(
            **report, by_split_family={f"{s}/{f}": n for (s, f), n in counts.items()},
            by_budget_group=dict(Counter(s["budget_group"] for s in metadata["samples"])),
        ))
        write(output / "stage.json", dict(stage="physics_distillation", time=time.time()))
        distillation = output / "distillation"
        command = [sys.executable, "-u", "-m", "fdtdmesh.physics", "distill",
                   "--manifest", str(MANIFEST), "--targets", str(targets),
                   "--initial-checkpoint", workflow["checkpoint"], "--output", str(distillation),
                   "--epochs", "20", "--batch-size", "4", "--learning-rate", "0.0003",
                   "--repair-weight", "0", "--sparse-sample-weight", "2",
                   "--projection-samples", "8", "--patience", "8", "--device", "cuda:0",
                   "--allow-dirty"]
        if (distillation / "resume.pt").exists():
            command.extend(["--resume", str(distillation / "resume.pt")])
        subprocess.run(command, cwd=ROOT, check=True)
        training = read(distillation / "training.json")
        write(output / "stage.json", dict(stage="complete", time=time.time(),
              epochs=len(training["history"]), best_validation_loss=training["best_validation_loss"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    reuse_from = args.reuse_from.resolve() if args.reuse_from else None
    if reuse_from is None and (output / "workflow.json").exists():
        recorded_source = read(output / "workflow.json").get("reuse_from")
        reuse_from = Path(recorded_source) if recorded_source else None
    if args.status:
        for name in ("stage.json", "progress.json", "failure.json"):
            if (output / name).exists():
                print(name, json.dumps(read(output / name), indent=2))
        return
    if args.worker is not None:
        worker(output, args.worker)
        return
    if args.foreground:
        try:
            run(output, reuse_from)
        except Exception as error:
            write(output / "failure.json", dict(error=str(error), type=type(error).__name__, time=time.time()))
            write(output / "stage.json", dict(stage="failed", time=time.time()))
            raise
        return
    output.mkdir(parents=True, exist_ok=True)
    # Refuse a duplicate coordinator before modifying its launch metadata.
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
    with (output / "workflow.log").open("a") as log:
        command = [sys.executable, "-u", __file__, "--foreground", "--output", str(output)]
        if reuse_from is not None:
            command.extend(["--reuse-from", str(reuse_from)])
        process = subprocess.Popen(
            command,
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(output / "launch.json", dict(pid=process.pid, time=time.time()))
    print(f"Physics campaign launched: PID {process.pid}; {output}")


if __name__ == "__main__":
    main()
