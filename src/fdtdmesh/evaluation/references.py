"""Resumable generation of converged FDTD reference observations."""

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from fdtdmesh.data.schema import SPLITS, provenance, read_manifest
from fdtdmesh.mesh import MESH_POLICY

from .metrics import spectrum
from .pipeline import EvaluationConfig, converge_reference, grids

REFERENCE_RUN_SCHEMA = 1


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _selection(splits, limit):
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Select one or more distinct splits")
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 1):
        raise ValueError("limit must be a positive integer")
    return splits, None if limit is None else int(limit)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_identity(manifest, config, splits, limit, scene_ids, reuse_roots):
    return {
        "schema_version": REFERENCE_RUN_SCHEMA,
        "dataset_id": manifest["dataset_id"],
        # Normalize tuples so the in-memory value exactly matches its JSON round trip.
        "config": json.loads(json.dumps(asdict(config), allow_nan=False)),
        "mesh_policy": MESH_POLICY,
        "selection": {
            "splits": list(splits),
            "limit_per_split": limit,
            "scene_ids": None if scene_ids is None else list(scene_ids),
        },
        "reuse_run_sha256": [_sha256(Path(root) / "run.json") for root in reuse_roots],
    }


def _load_completed(directory, spec):
    status_path = directory / "reference.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("scene_id") != spec.scene_id or status.get("scene_hash") != spec.content_hash:
        raise ValueError(f"Existing reference identity mismatch for {spec.scene_id}")
    scene_path = directory / "scene.json"
    evaluated_path = directory / "evaluated_scene.json"
    if not scene_path.exists() or not evaluated_path.exists():
        raise ValueError(f"Committed reference is incomplete for {spec.scene_id}")
    if json.loads(scene_path.read_text(encoding="utf-8")) != spec.to_dict():
        raise ValueError(f"Stored scene differs from manifest for {spec.scene_id}")
    evaluated = json.loads(evaluated_path.read_text(encoding="utf-8"))
    if evaluated.get("t_end") != status.get("duration"):
        raise ValueError(f"Stored duration differs from status for {spec.scene_id}")
    if status.get("status") == "converged" and not (directory / "reference_latest.npz").exists():
        raise ValueError(f"Converged reference data is missing for {spec.scene_id}")
    return status


def _coverage(scenes):
    states = ("converged", "nonconverged", "time_unsettled", "failed")
    counts = {state: 0 for state in states}
    by_split = {}
    for row in scenes:
        state = row["status"]
        counts[state] = counts.get(state, 0) + 1
        split_counts = by_split.setdefault(row["split"], {name: 0 for name in states})
        split_counts[state] = split_counts.get(state, 0) + 1
    return {
        "completed": len(scenes),
        "accepted": counts["converged"],
        "counts": counts,
        "by_split": by_split,
    }


def _reuse_completed(directory, spec, reuse_roots):
    for root in reuse_roots:
        source = Path(root) / spec.scene_id
        try:
            status = _load_completed(source, spec)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if status is None or status.get("status") != "converged":
            continue
        for name in (
            "scene.json",
            "evaluated_scene.json",
            "reference_latest.npz",
        ):
            shutil.copy2(source / name, directory / name)
        reused = {
            **status,
            "reused": True,
            "reused_from_dataset_id": json.loads(
                (Path(root) / "run.json").read_text(encoding="utf-8")
            )["dataset_id"],
        }
        _write_json(directory / "reference.json", reused)
        return reused
    return None


def _write_summary(output, report):
    coverage = report["coverage"]
    lines = [
        "# Converged reference generation",
        "",
        f"Completed: {coverage['completed']}/{report['selected_scene_count']} scenes.",
        f"Accepted: {coverage['accepted']} converged references.",
        "",
        "| Split | Converged | Nonconverged | Time unsettled | Failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in report["selection"]["splits"]:
        row = coverage["by_split"].get(split, {})
        lines.append(
            f"| {split} | {row.get('converged', 0)} | {row.get('nonconverged', 0)} | "
            f"{row.get('time_unsettled', 0)} | {row.get('failed', 0)} |"
        )
    lines += [
        "",
        "Only `converged` scenes are valid ground truth. Every other status is retained.",
        "Duration extensions restart the spatial sequence so all levels use one physical window.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_references(
    manifest_path,
    output,
    *,
    config=None,
    splits=("train", "validation", "test_iid"),
    limit=None,
    scene_ids=None,
    reuse_roots=(),
    runner=None,
):
    """Generate or resume per-scene references, committing each result atomically."""
    config = config or EvaluationConfig()
    splits, limit = _selection(splits, limit)
    manifest, all_scenes = read_manifest(manifest_path)
    if scene_ids is not None:
        scene_ids = tuple(scene_ids)
        scene_map = {scene.scene_id: scene for scene in all_scenes}
        if len(set(scene_ids)) != len(scene_ids) or not all(
            scene_id in scene_map for scene_id in scene_ids
        ):
            raise ValueError("Every selected scene ID must exist exactly once")
        selected = [scene_map[scene_id] for scene_id in scene_ids]
    else:
        selected = []
        for split in splits:
            candidates = [scene for scene in all_scenes if scene.split == split]
            selected.extend(candidates if limit is None else candidates[:limit])
    if not selected:
        raise ValueError("No scenes selected")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reuse_roots = tuple(Path(root) for root in reuse_roots)
    if any(not (root / "run.json").is_file() for root in reuse_roots):
        raise ValueError("Every reuse root must contain run.json")
    identity = _run_identity(manifest, config, splits, limit, scene_ids, reuse_roots)
    run_path = output / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("Reference output belongs to a different dataset or configuration")
    else:
        unexpected = [path for path in output.iterdir() if path.name != "run.json.tmp"]
        if unexpected:
            raise ValueError("Reference output has files but no compatible run identity")
        _write_json(run_path, identity)

    report = {
        **identity,
        "provenance": provenance(),
        "selected_scene_count": len(selected),
        "scenes": [],
    }
    for spec in selected:
        directory = output / spec.scene_id
        directory.mkdir(exist_ok=True)
        status = _load_completed(directory, spec)
        if status is None:
            status = _reuse_completed(directory, spec, reuse_roots)
        if status is None:
            print(f"Reference {spec.scene_id}", flush=True)
            started = perf_counter()
            if runner is None:
                status, latest = converge_reference(spec, config)
            else:
                status, latest = converge_reference(spec, config, runner=runner)
            status = {
                **status,
                "scene_id": spec.scene_id,
                "scene_hash": spec.content_hash,
                "split": spec.split,
                "family": spec.family,
                "wall_seconds": perf_counter() - started,
            }
            evaluated = replace(spec, t_end=status.get("duration", spec.t_end))
            _write_json(directory / "scene.json", spec.to_dict())
            _write_json(directory / "evaluated_scene.json", evaluated.to_dict())
            if latest is not None:
                times, frequencies = grids(evaluated, config)
                result, observations = latest
                np.savez_compressed(
                    directory / "reference_latest.npz",
                    times=times,
                    frequencies=frequencies,
                    waveforms=observations,
                    spectra=spectrum(observations, times, frequencies),
                    x=result.mesh.x,
                    y=result.mesh.y,
                    receiver_coordinates=np.concatenate(list(result.receiver_coordinates.values())),
                )
            _write_json(directory / "reference.json", status)
        else:
            print(f"Reference {spec.scene_id}: resume {status['status']}", flush=True)
        report["scenes"].append(status)
        report["coverage"] = _coverage(report["scenes"])
        _write_json(output / "report.json", report)
        _write_summary(output, report)
    return report
