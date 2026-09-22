"""Quota-driven, restart-safe multi-GPU generation of strict v6 references.

Run ``python -m fdtdmesh.data.campaign --help``. Every attempted scene, including
rejects, has a stable identity and immutable per-scene manifest. A final manifest
is published only after all split/interaction quotas have been met.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from fdtdmesh.data.generate_v6 import GENERATOR_VERSION, RichConfig, feature_audit, make_rich_scene
from fdtdmesh.data.schema import SceneSpec, digest, provenance, write_manifest
from fdtdmesh.evaluation import EvaluationConfig, generate_references
from fdtdmesh.evaluation.references import _write_json

REFERENCE_LEVELS = (128, 256, 512, 1024, 2048, 4096)


def reference_config(max_reference_level=4096, *, extend_nonconverged=True):
    if max_reference_level not in REFERENCE_LEVELS or max_reference_level < 1024:
        raise ValueError("Maximum reference level must be 1024, 2048, or 4096")
    return EvaluationConfig(
        reference_levels=tuple(level for level in REFERENCE_LEVELS if level <= max_reference_level),
        minimum_reference_level=1024,
        relative_tolerance=0.02,
        consecutive_passes=2,
        tail_relative_tolerance=0.01,
        max_duration_extensions=6,
        extend_nonconverged=extend_nonconverged,
        max_cell_updates=20_000_000_000_000,
        max_field_bytes=6_000_000_000,
        max_history_bytes=1_000_000_000,
        max_observation_samples=262145,
        max_frequency_samples=32769,
        wavelength_cells=16,
        attenuation_cells=4,
    )


def quotas(target, workers, lane, *, balanced=False):
    """Spread each split across lanes and four topology classes, with exact totals."""
    if balanced:
        totals = [target // 4 + (family < target % 4) for family in range(4)]
        return [
            n // workers + ((lane - family) % workers < n % workers)
            for family, n in enumerate(totals)
        ]
    per_lane = target // workers + (lane < target % workers)
    return [per_lane // 4 + (family < per_lane % 4) for family in range(4)]


def summarize(output):
    rows = []
    for path in sorted(Path(output).glob("lane*/decisions/*.json")):
        rows.append(json.loads(path.read_text()))
    counts = Counter(r["status"] for r in rows)
    by_split = {
        split: dict(Counter(r["status"] for r in rows if r["split"] == split))
        for split in ("train", "validation", "test_iid")
    }
    return dict(
        completed=len(rows),
        accepted=counts.get("converged", 0),
        counts=dict(counts),
        by_split=by_split,
        decisions=rows,
    )


def _attempt(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    generate_references(
        directory / "manifest.json",
        directory / "references",
        config=EvaluationConfig(**config),
        record_progress=True,
    )


def _worker(output, lane):
    output = Path(output).resolve()
    config = json.loads((output / "campaign.json").read_text())
    version = config.get("generator_version", GENERATOR_VERSION)
    if version == 7:
        from .generate_v7 import SparseConfig, make_sparse_scene
        from .generate_v7 import feature_audit as audit_scene

        generate_scene, generator_config = make_sparse_scene, SparseConfig(**config["generator"])
    elif version == 6:
        generate_scene, generator_config = make_rich_scene, RichConfig(**config["generator"])
        audit_scene = feature_audit
    else:
        raise ValueError("Unsupported campaign generator")
    folder = output / f"lane{lane}"
    (folder / "decisions").mkdir(parents=True, exist_ok=True)
    accepted = Counter()
    for path in (folder / "decisions").glob("*.json"):
        row = json.loads(path.read_text())
        if row["status"] == "converged":
            accepted[(row["split"], row["family_index"])] += 1
    for split, target in config["targets"].items():
        required = quotas(target, len(config["gpus"]), lane, balanced=version == 7)
        for local_index in range(config["max_candidates_per_split_per_lane"]):
            if all(accepted[(split, k)] >= required[k] for k in range(4)):
                break
            family = local_index % 4
            index = lane * 1_000_000 + local_index
            scene_id = f"{split}-v{version}-{index:06d}"
            decision_path = folder / "decisions" / f"{scene_id}.json"
            if decision_path.exists() or accepted[(split, family)] >= required[family]:
                continue
            directory = folder / "attempts" / scene_id
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            started = time.time()
            if not manifest_path.exists():
                try:
                    scene = generate_scene(config["seed"], split, index, config=generator_config)
                    if version == 7:
                        from .supplement import reserve_geometry

                        if not reserve_geometry(output, scene):
                            raise RuntimeError("Near-duplicate geometry reserved in another split")
                except RuntimeError as error:
                    _write_json(
                        decision_path,
                        dict(
                            scene_id=scene_id,
                            split=split,
                            family_index=family,
                            status="generation_rejected",
                            reason=str(error),
                        ),
                    )
                    continue
                audit = audit_scene(scene, generator_config)
                write_manifest(
                    manifest_path,
                    [scene],
                    generation=dict(
                        version=version,
                        seed=config["seed"],
                        config=config["generator"],
                        campaign_id=config["campaign_id"],
                        feature_audit=audit,
                        layout_attempts=scene.generation_attempts,
                    ),
                )
            record = json.loads(manifest_path.read_text())["scenes"][0]
            # Resume the exact per-scene convergence contract after an interruption.
            _write_json(directory / "config.json", config["reference"])
            command = [
                sys.executable,
                "-u",
                "-m",
                "fdtdmesh.data.campaign",
                "--attempt",
                str(directory),
            ]
            _write_json(
                folder / "active.json",
                dict(
                    scene_id=scene_id,
                    command=command,
                    started=started,
                    wall_limit_seconds=config["scene_wall_seconds"],
                ),
            )
            with (directory / "worker.log").open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=config["scene_wall_seconds"])
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = None
            path = directory / "references" / scene_id / "reference.json"
            if path.exists():
                status = json.loads(path.read_text())
                state = status["status"]
                # Distinguish exhausted numerical/resource policy from infrastructure faults.
                if state == "failed":
                    errors = [
                        entry.get("error", "")
                        for attempt in status.get("duration_history", [])
                        for entry in attempt.get("levels", [])
                    ]
                    errors.append(status.get("error", ""))
                    resource = any("resource limit" in error.lower() for error in errors)
                    if not resource:
                        raise RuntimeError(f"Unexpected solver failure in {scene_id}; see {path}")
                    state = "nonconverged"
                if state == "time_unsettled":
                    state = "nonconverged"
                reason = status["status"]
            elif code is None:
                state, reason = "nonconverged", "hard_wall_time_limit"
            else:
                raise RuntimeError(f"Worker exited {code} without a committed record: {directory}")
            row = dict(
                scene_id=scene_id,
                scene_hash=digest(record),
                split=split,
                family_index=family,
                status=state,
                terminal_reason=reason,
                wall_seconds=time.time() - started,
                manifest=str(manifest_path.relative_to(output)),
                reference=str(path.relative_to(output)) if path.exists() else None,
            )
            _write_json(decision_path, row)
            if state == "converged":
                accepted[(split, family)] += 1
            print(json.dumps(row), flush=True)
        if not all(accepted[(split, k)] >= required[k] for k in range(4)):
            raise RuntimeError(
                f"Candidate safety limit exhausted for {split}, lane {lane}; quota unmet"
            )
    _write_json(
        folder / "complete.json", dict(accepted={f"{s}/{f}": n for (s, f), n in accepted.items()})
    )


def finalize(output, config):
    """Publish accepted records only after quota, artifact and cross-split checks."""
    summary = summarize(output)
    scenes = []
    for row in summary["decisions"]:
        if row["status"] != "converged":
            continue
        manifest = json.loads((output / row["manifest"]).read_text())
        scene = SceneSpec.from_dict(manifest["scenes"][0])
        path = output / row["reference"]
        status = json.loads(path.read_text())
        if status["scene_hash"] != scene.content_hash or status["status"] != "converged":
            raise ValueError("Reference identity/acceptance mismatch")
        if not (path.parent / "reference_latest.npz").exists():
            raise ValueError("Accepted waveform missing")
        scenes.append(scene)
    if any(sum(s.split == split for s in scenes) < n for split, n in config["targets"].items()):
        raise ValueError("Cannot publish: accepted quota unmet")
    # write_manifest applies scene lineage and near-geometry cross-split checks.
    manifest = write_manifest(
        output / "accepted_manifest.json",
        scenes,
        generation=dict(
            version=config.get("generator_version", GENERATOR_VERSION),
            campaign_id=config["campaign_id"],
            config=config["generator"],
            selection="strictly_converged_with_explicit_topology_and_split_quotas",
        ),
    )
    _write_json(
        output / "complete.json",
        dict(dataset_id=manifest["dataset_id"], accepted=len(scenes), finished_unix=time.time()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--generator-version", type=int, choices=(6, 7), default=6)
    parser.add_argument("--scene-wall-seconds", type=int, default=21600)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--material-averaging", choices=["point", "sampled"], default="sampled")
    parser.add_argument("--averaging-samples", type=int, default=8)
    parser.add_argument("--averaging-max-samples", type=int, default=32)
    parser.add_argument("--averaging-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--max-reference-level",
        type=int,
        choices=(1024, 2048, 4096),
        default=4096,
        help="Mark a scene nonconverged if it still fails after this level",
    )
    parser.add_argument(
        "--extend-nonconverged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry spatially nonconverged scenes with a longer time window",
    )
    parser.add_argument("--worker", type=int)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.attempt is not None:
        return _attempt(args.attempt)
    if args.output is None:
        parser.error("--output is required")
    if args.worker is not None:
        return _worker(args.output, args.worker)
    if (
        min(args.train, args.validation, args.test, args.scene_wall_seconds, args.max_candidates)
        <= 0
    ):
        parser.error("Quotas and limits must be positive")
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU IDs must be distinct and nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Linux production worker: lifetime lock prevents concurrent/resumed coordinators.
    import fcntl

    lock = (output / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stop(signum, frame):
        raise SystemExit(f"Coordinator stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    prov = provenance()
    config = dict(
        version=1,
        gpus=args.gpus,
        seed=args.seed,
        targets=dict(train=args.train, validation=args.validation, test_iid=args.test),
        generator=asdict(RichConfig()),
        reference=json.loads(
            json.dumps(
                asdict(
                    replace(
                        reference_config(
                            args.max_reference_level,
                            extend_nonconverged=args.extend_nonconverged,
                        ),
                        material_averaging=args.material_averaging,
                        averaging_samples=args.averaging_samples,
                        averaging_max_samples=args.averaging_max_samples,
                        averaging_tolerance=args.averaging_tolerance,
                    )
                )
            )
        ),
        scene_wall_seconds=args.scene_wall_seconds,
        max_candidates_per_split_per_lane=args.max_candidates,
        source_sha256=prov["source_sha256"],
        native_binaries=prov["native_binaries"],
    )
    if args.generator_version == 7:
        from .generate_v7 import SparseConfig

        config.update(generator_version=7, generator=asdict(SparseConfig()))
    config["campaign_id"] = digest(config)
    path = output / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Campaign config/source/native identity differs; use a new output")
    _write_json(path, config)
    _write_json(output / "provenance.json", prov)
    (output / "coordinator.pid").write_text(str(os.getpid()))
    processes = []
    try:
        for lane, gpu in enumerate(args.gpus):
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            )
            with (output / f"lane{lane}.log").open("a") as log:
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "fdtdmesh.data.campaign",
                        "--output",
                        str(output),
                        "--worker",
                        str(lane),
                    ],
                    start_new_session=True,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            processes.append(p)
        _write_json(
            output / "workers.json",
            [dict(lane=i, gpu=g, pid=p.pid) for i, (g, p) in enumerate(zip(args.gpus, processes))],
        )
        while True:
            summary = summarize(output)
            _write_json(
                output / "progress.json",
                dict(
                    campaign_id=config["campaign_id"],
                    targets=config["targets"],
                    updated_unix=time.time(),
                    **{k: v for k, v in summary.items() if k != "decisions"},
                ),
            )
            if all(p.poll() is not None for p in processes):
                break
            time.sleep(20)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("A worker stopped before its quota; inspect lane logs and resume")
        finalize(output, config)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
