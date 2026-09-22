"""Run a balanced v6 physics-search campaign and distill its selected targets."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fdtdmesh.data.schema import provenance, read_manifest
from fdtdmesh.mesh import MESH_POLICY
from fdtdmesh.physics import PHYSICS_TARGET_VERSION
from fdtdmesh.training import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("separated", "contact", "overlap", "nested")
BUDGETS = (48, 64, 96, 128)
PILOT_VALIDATION_SCENES = {
    "validation-v6-000000",
    "validation-v6-000002",
    "validation-v6-000005",
    "validation-v6-000007",
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _select_scenes(manifest_path, teacher_metadata_path):
    _, scenes = read_manifest(manifest_path)
    metadata = json.loads(Path(teacher_metadata_path).read_text(encoding="utf-8"))
    feasible = defaultdict(set)
    for sample in metadata["samples"]:
        feasible[sample["scene_id"]].add(tuple(sample["budget"]))
    required = {(budget, budget) for budget in BUDGETS}
    families = list(FAMILIES) + sorted(
        {s.family for s in scenes if s.split == "train"} - set(FAMILIES)
    )
    families = [f for f in families if any(s.family == f and s.split == "train" for s in scenes)]
    selected = {"train": {}, "validation": {}}
    counts = {"train": 16, "validation": 4}
    for split in selected:
        for family in families:
            candidates = sorted(
                scene.scene_id
                for scene in scenes
                if scene.split == split
                and scene.family == family
                and required <= feasible[scene.scene_id]
                and scene.scene_id not in PILOT_VALIDATION_SCENES
            )
            if len(candidates) < counts[split]:
                raise ValueError(f"Insufficient feasible {split}/{family} scenes")
            selected[split][family] = candidates[: counts[split]]
    lanes = [[] for _ in range(4)]
    for split in ("train", "validation"):
        for family in families:
            for index, scene_id in enumerate(selected[split][family]):
                lanes[index % 4].append(scene_id)
    return selected, lanes


def _prepare_references(output, dataset_id, scene_ids, reference_audit_path):
    reference_root = output / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(reference_audit_path).read_text(encoding="utf-8"))
    if audit["dataset_id"] != dataset_id:
        raise ValueError("Reference audit belongs to a different dataset")
    run = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_audit": str(Path(reference_audit_path).resolve()),
        "scene_ids": sorted(scene_ids),
    }
    run_path = reference_root / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")) != run:
        raise ValueError("Reference binding changed; choose a new output")
    _write_json(run_path, run)
    for scene_id in scene_ids:
        link = reference_root / scene_id
        target = Path(audit["references"][scene_id])
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Reference link changed for {scene_id}")
        elif link.exists():
            raise ValueError(f"Reference path is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    return reference_root


def _search_command(args, reference_root, lane_output, scene_ids):
    return [
        sys.executable,
        "-u",
        "-m",
        "fdtdmesh.physics",
        "search",
        "--manifest",
        str(args.manifest),
        "--teacher-targets",
        str(args.teacher_targets),
        "--checkpoint",
        str(args.checkpoint),
        "--references",
        str(reference_root),
        "--output",
        str(lane_output),
        "--splits",
        "train",
        "validation",
        "--scenes",
        *scene_ids,
        "--budgets",
        *map(str, BUDGETS),
        "--beta",
        "0.02",
        "--physics-weight",
        "0.5",
        "--perturbations",
        "2",
        "--meshing-time-limit",
        "30",
        "--material-averaging",
        "sampled",
        "--averaging-samples",
        "8",
        "--averaging-max-samples",
        "32",
        "--averaging-tolerance",
        "0.001",
        "--device",
        "cuda:0",
    ]


def _merge_targets(manifest, lane_outputs, output):
    merged = output / "merged"
    merged.mkdir(exist_ok=True)
    arrays_path = merged / "physics_targets.npz"
    metadata_path = merged / "physics_targets.json"
    samples, target_x, target_y, lane_identities = [], [], [], []
    seen = set()
    for lane_output in lane_outputs:
        metadata = json.loads((lane_output / "physics_targets.json").read_text(encoding="utf-8"))
        with np.load(lane_output / "physics_targets.npz") as arrays:
            x, y = arrays["target_x"].copy(), arrays["target_y"].copy()
        if len(x) != len(metadata["samples"]) or len(y) != len(metadata["samples"]):
            raise ValueError(f"Target count mismatch in {lane_output}")
        for index, sample in enumerate(metadata["samples"]):
            key = (sample["scene_id"], tuple(sample["budget"]))
            if key in seen:
                raise ValueError(f"Duplicate physics target {key}")
            seen.add(key)
            samples.append(sample)
            target_x.append(x[index])
            target_y.append(y[index])
        lane_identities.append(metadata["search_identity"])
    temporary = merged / "physics_targets.tmp.npz"
    np.savez_compressed(
        temporary,
        target_x=np.asarray(target_x, dtype=np.float32),
        target_y=np.asarray(target_y, dtype=np.float32),
    )
    temporary.replace(arrays_path)
    metadata = {
        "schema_version": 1,
        "physics_target_version": PHYSICS_TARGET_VERSION,
        "dataset_id": manifest["dataset_id"],
        "targets_sha256": _sha256(arrays_path),
        "mesh_policy": MESH_POLICY,
        "raster_shape": [128, 128],
        "samples": samples,
        "search_identity": {"lanes": lane_identities},
        "provenance": provenance(),
    }
    _write_json(metadata_path, metadata)
    return arrays_path, metadata


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, _ = read_manifest(args.manifest)
        selected, lanes = _select_scenes(args.manifest, args.teacher_targets.with_suffix(".json"))
        scene_ids = [scene_id for lane in lanes for scene_id in lane]
        reference_root = _prepare_references(
            output, manifest["dataset_id"], scene_ids, args.reference_audit
        )
        distill_config = TrainingConfig(
            epochs=20,
            batch_size=4,
            learning_rate=3e-4,
            width=16,
            repair_weight=0.05,
            repair_every=4,
            projection_samples=8,
            seed=2026,
            early_stopping_patience=8,
            sparse_sample_weight=2.0,
        )
        identity = {
            "dataset_id": manifest["dataset_id"],
            "manifest": str(args.manifest),
            "teacher_targets_sha256": _sha256(args.teacher_targets),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "budgets": list(BUDGETS),
            "selection": selected,
            "lanes": lanes,
            "search": {"beta": 0.02, "physics_weight": 0.5, "perturbations": 2},
            "distillation": distill_config.__dict__,
            "source_sha256": provenance()["source_sha256"],
        }
        workflow_path = output / "workflow.json"
        if (
            workflow_path.exists()
            and json.loads(workflow_path.read_text(encoding="utf-8")) != identity
        ):
            raise ValueError("Physics campaign identity changed; choose a new output")
        _write_json(workflow_path, identity)
        _write_json(output / "stage.json", {"stage": "physics_search", "time": time.time()})
        lane_outputs, processes, logs = [], [], []
        for lane, lane_scenes in enumerate(lanes):
            lane_root = output / f"lane{lane}"
            lane_output = lane_root / "search"
            lane_root.mkdir(exist_ok=True)
            command = _search_command(args, reference_root, lane_output, lane_scenes)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(lane),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
            log = (lane_root / "search.log").open("a")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            lane_outputs.append(lane_output)
            processes.append(process)
            logs.append(log)
        try:
            return_codes = [process.wait() for process in processes]
        finally:
            for log in logs:
                log.close()
        if any(return_codes):
            raise RuntimeError(f"Physics search lane failures: {return_codes}")
        _write_json(output / "stage.json", {"stage": "merge_targets", "time": time.time()})
        targets, metadata = _merge_targets(manifest, lane_outputs, output)
        split_counts = defaultdict(int)
        for sample in metadata["samples"]:
            split_counts[sample["split"]] += 1
        expected_counts = {
            split: sum(len(ids) for ids in groups.values()) * len(BUDGETS)
            for split, groups in selected.items()
        }
        if split_counts != expected_counts:
            raise ValueError(f"Unexpected merged target counts: {dict(split_counts)}")
        _write_json(
            output / "stage.json",
            {"stage": "physics_distillation", "targets": dict(split_counts), "time": time.time()},
        )
        distillation = output / "distillation"
        command = [
            sys.executable,
            "-u",
            "-m",
            "fdtdmesh.physics",
            "distill",
            "--manifest",
            str(args.manifest),
            "--targets",
            str(targets),
            "--initial-checkpoint",
            str(args.checkpoint),
            "--output",
            str(distillation),
            "--epochs",
            "20",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.0003",
            "--repair-weight",
            "0.05",
            "--repair-every",
            "4",
            "--sparse-sample-weight",
            str(distill_config.sparse_sample_weight),
            "--projection-samples",
            "8",
            "--patience",
            "8",
            "--device",
            "cuda:0",
            "--allow-dirty",
        ]
        resume = distillation / "resume.pt"
        if resume.exists():
            command.extend(("--resume", str(resume)))
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError(f"Physics distillation failed with code {result.returncode}")
        training = json.loads((distillation / "training.json").read_text(encoding="utf-8"))
        _write_json(
            output / "stage.json",
            {
                "stage": "complete",
                "epochs": len(training["history"]),
                "best_validation_loss": training["best_validation_loss"],
                "time": time.time(),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/reference_v6_2000/accepted_manifest.json"),
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=Path("artifacts/training_v6_2000/teacher/targets.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/training_v6_2000/pretraining/best.pt"),
    )
    parser.add_argument(
        "--reference-audit",
        type=Path,
        default=Path("artifacts/training_v6_2000/reference_audit.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/physics_distillation_v6"))
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    for name in ("manifest", "teacher_targets", "checkpoint", "reference_audit", "output"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)
    if args.foreground:
        try:
            run(args)
        except Exception as error:
            _write_json(
                args.output / "failure.json",
                {"error": str(error), "error_type": type(error).__name__, "time": time.time()},
            )
            raise
        return
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--foreground",
        "--manifest",
        str(args.manifest),
        "--teacher-targets",
        str(args.teacher_targets),
        "--checkpoint",
        str(args.checkpoint),
        "--reference-audit",
        str(args.reference_audit),
        "--output",
        str(args.output),
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
    with (args.output / "workflow.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(
        args.output / "launch.json",
        {"pid": process.pid, "command": command, "time": time.time()},
    )
    print(f"Physics-distillation workflow PID {process.pid}; output {args.output}")


if __name__ == "__main__":
    main()
