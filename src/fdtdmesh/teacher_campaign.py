"""Resumable, CPU-parallel legal teacher targets for large accepted corpora."""

import fcntl
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fdtdmesh.budget_schedule import normalize_budgets
from fdtdmesh.data.schema import SceneSpec, digest, provenance, read_manifest
from fdtdmesh.evaluation.references import _write_json
from fdtdmesh.mesh import MESH_POLICY
from fdtdmesh.training import (
    TEACHER_VERSION,
    _sha256,
    load_teacher_targets,
    teacher_sample,
)


def _job(record, budget, time_limit):
    scene = SceneSpec.from_dict(record)
    started = time.time()
    try:
        sample, x, y = teacher_sample(scene, budget, scene.raster_shape, time_limit)
        return dict(sample=sample, x=x.tolist(), y=y.tolist(), seconds=time.time() - started)
    except (ValueError, RuntimeError) as error:
        return dict(
            failure=dict(
                scene_id=scene.scene_id,
                scene_hash=scene.content_hash,
                split=scene.split,
                budget=list(budget),
                error=str(error),
                error_type=type(error).__name__,
            ),
            seconds=time.time() - started,
        )


def prepare_targets(
    manifest_path,
    output,
    *,
    workers=12,
    budgets=(48, 64, 96, 128),
    time_limit=30.0,
    budget_plan=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "targets.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, scenes = read_manifest(manifest_path)
        scenes = [scene for scene in scenes if scene.split in ("train", "validation")]
        shape = {tuple(scene.raster_shape) for scene in scenes}
        if len(shape) != 1:
            raise ValueError("Expected a common raster shape")
        shape = shape.pop()
        budgets = normalize_budgets(budgets)
        if budget_plan is not None:
            if set(budget_plan["assignments"]) != {s.scene_id for s in scenes}:
                raise ValueError("Budget plan must cover exactly train and validation scenes")
            for entries in budget_plan["assignments"].values():
                normalize_budgets([entry["budget"] for entry in entries])
        prov = provenance()
        identity = dict(
            dataset_id=manifest["dataset_id"],
            manifest_sha256=_sha256(manifest_path),
            budgets=[list(b) for b in budgets],
            time_limit=time_limit,
            source_sha256=prov["source_sha256"],
            teacher_version=TEACHER_VERSION,
            mesh_policy=MESH_POLICY,
            splits=["train", "validation"],
        )
        if budget_plan is not None:
            identity["budget_plan_sha256"] = digest(budget_plan)
            identity["budgets"] = None
        identity_path = output / "targets_run.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Target campaign identity changed; choose a new output")
        _write_json(identity_path, identity)
        target_path = output / "targets.npz"
        if target_path.exists() and target_path.with_suffix(".json").exists():
            load_teacher_targets(manifest_path, target_path)
            return target_path
        cache = output / "target_records"
        cache.mkdir(exist_ok=True)
        jobs, groups = [], {}
        for scene in scenes:
            entries = (
                budget_plan["assignments"][scene.scene_id]
                if budget_plan is not None
                else [dict(budget=b, group="unspecified") for b in budgets]
            )
            for entry in entries:
                b = tuple(entry["budget"])
                path = cache / f"{scene.scene_id}-{b[0]}x{b[1]}.json"
                jobs.append((scene, b, path))
                groups[path.name] = entry["group"]
        results = {}
        pending = []
        for scene, budget, path in jobs:
            if path.exists():
                value = json.loads(path.read_text())
                record = value.get("sample", value.get("failure"))
                if (
                    record["scene_hash"] != scene.content_hash
                    or record["budget"] != list(budget)
                    or record.get("budget_group", "unspecified") != groups[path.name]
                ):
                    raise ValueError(f"Cached target identity differs: {path}")
                results[path.name] = value
            else:
                pending.append((scene, budget, path))
        started = time.time()

        def record_progress():
            status = dict(
                stage="teacher_targets",
                total=len(jobs),
                completed=len(results),
                feasible=sum("sample" in v for v in results.values()),
                failures=sum("failure" in v for v in results.values()),
                elapsed_seconds=time.time() - started,
                updated_unix=time.time(),
            )
            _write_json(output / "progress.json", status)
            return status

        record_progress()
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(_job, s.to_dict(), b, time_limit): path for s, b, path in pending
            }
            for future in as_completed(futures):
                path = futures[future]
                value = future.result()
                value.get("sample", value.get("failure"))["budget_group"] = groups[path.name]
                _write_json(path, value)
                results[path.name] = value
                status = record_progress()
                if len(results) % 100 == 0:
                    print(json.dumps(status), flush=True)
        values = [results[path.name] for _, _, path in jobs]
        coverage = {}
        for (scene, budget, path), value in zip(jobs, values):
            key = f"{scene.split}/{scene.family}/{groups[path.name]}"
            row = coverage.setdefault(key, dict(requested=0, feasible=0, failures=0))
            row["requested"] += 1
            row["feasible" if "sample" in value else "failures"] += 1
        _write_json(output / "coverage.json", coverage)
        accepted = [v for v in values if "sample" in v]
        failures = [v["failure"] for v in values if "failure" in v]
        if not accepted or {v["sample"]["split"] for v in accepted} != {"train", "validation"}:
            raise ValueError("Both train and validation need feasible teacher targets")
        temporary = output / "targets.tmp.npz"
        np.savez_compressed(
            temporary,
            target_x=np.asarray([v["x"] for v in accepted], dtype=np.float32),
            target_y=np.asarray([v["y"] for v in accepted], dtype=np.float32),
        )
        temporary.replace(target_path)
        metadata = dict(
            schema_version=1,
            teacher_version=TEACHER_VERSION,
            dataset_id=manifest["dataset_id"],
            manifest_sha256=_sha256(manifest_path),
            targets_sha256=_sha256(target_path),
            mesh_policy=MESH_POLICY,
            raster_shape=list(shape),
            splits=["train", "validation"],
            budget_override=[list(b) for b in budgets] if budget_plan is None else None,
            budget_plan_sha256=None if budget_plan is None else digest(budget_plan),
            teacher="material_edge_heuristic_projected_to_legal_mesh_then_rebinned",
            samples=[v["sample"] for v in accepted],
            failures=failures,
            coverage=coverage,
            seconds=sum(v["seconds"] for v in values),
            provenance=prov,
        )
        _write_json(target_path.with_suffix(".json"), metadata)
        print(
            f"Teacher targets ready: {len(accepted)} feasible, {len(failures)} failures", flush=True
        )
        return target_path
