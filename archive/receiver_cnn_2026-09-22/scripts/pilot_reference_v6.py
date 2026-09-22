"""Run a bounded pilot of v6 training candidates with the production convergence policy."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from fdtdmesh.data.campaign import reference_config
from fdtdmesh.data.generate_v6 import RichConfig, make_rich_scene
from fdtdmesh.data.schema import provenance, read_manifest, write_manifest
from fdtdmesh.evaluation.references import _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Use an already validated v6 manifest containing the requested indices"
    )
    parser.add_argument("--count", type=int, default=4, choices=range(1, 11))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pilot.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    supplied = {}
    manifest_id = None
    if args.manifest is not None:
        manifest, scenes = read_manifest(args.manifest)
        supplied = {s.scene_id: s for s in scenes}
        if not {f"train-v6-{i:06d}" for i in range(args.count)} <= set(supplied):
            raise ValueError(
                "Pilot manifest is missing requested training scenes"
            )
        if any(s.pec_policy != "rectangles_and_wires" for s in scenes):
            raise ValueError("Pilot manifest must use the v6 PEC/anchor contract")
        if manifest.get("generation", {}).get("config") != asdict(RichConfig()):
            raise ValueError("Pilot manifest generator configuration differs")
        manifest_id = manifest["dataset_id"]
    config = dict(
        seed=2026,
        indices=list(range(args.count)),
        gpus=list(range(4)),
        scene_wall_seconds=21600,
        generator=asdict(RichConfig()),
        reference=asdict(replace(reference_config(), material_averaging="sampled")),
        provenance=provenance(),
        supplied_dataset_id=manifest_id,
    )
    config = json.loads(json.dumps(config))
    identity = output / "pilot.json"
    if identity.exists() and json.loads(identity.read_text()) != config:
        raise ValueError("Pilot identity changed; choose a new output directory")
    _write_json(identity, config)
    (output / "coordinator.pid").write_text(str(os.getpid()))
    started = time.time()

    def lane(gpu):
        rows = []
        for index in range(gpu, args.count, 4):
            directory = output / f"train-v6-{index:06d}"
            directory.mkdir(exist_ok=True)
            decision = directory / "decision.json"
            if decision.exists():
                rows.append(json.loads(decision.read_text()))
                continue
            begin = time.time()
            scene = supplied.get(f"train-v6-{index:06d}")
            if scene is None:
                scene = make_rich_scene(2026, "train", index)
            write_manifest(
                directory / "manifest.json",
                [scene],
                generation=dict(
                    version=6, purpose="bounded_candidate_pilot", config=config["generator"]
                ),
            )
            _write_json(directory / "config.json", config["reference"])
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "fdtdmesh.data.campaign",
                "--attempt",
                str(directory),
            ]
            with (directory / "worker.log").open("a") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                _write_json(
                    directory / "process.json",
                    dict(pid=process.pid, gpu=gpu, started_unix=begin, command=command),
                )
                timeout = False
                try:
                    code = process.wait(timeout=config["scene_wall_seconds"])
                except subprocess.TimeoutExpired:
                    timeout = True
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = process.returncode
            reference = directory / "references" / scene.scene_id / "reference.json"
            status = json.loads(reference.read_text()) if reference.exists() else {}
            row = dict(
                scene_id=scene.scene_id,
                family=scene.family,
                gpu=gpu,
                status=status.get("status", "hard_wall_time_limit" if timeout else "failed"),
                accepted_budget=status.get("accepted_budget"),
                duration_extensions=max(0, len(status.get("duration_history", [])) - 1),
                simulation_duration=status.get("duration"),
                reference_wall_seconds=status.get("wall_seconds"),
                total_wall_seconds=time.time() - begin,
                returncode=code,
                reference=str(reference) if reference.exists() else None,
            )
            _write_json(decision, row)
            print(json.dumps(row), flush=True)
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for lane_rows in pool.map(lane, range(4)) for row in lane_rows]
    rows.sort(key=lambda row: row["scene_id"])
    elapsed = time.time() - started
    summary = dict(
        completed=len(rows),
        accepted=sum(r["status"] == "converged" for r in rows),
        elapsed_seconds=elapsed,
        mean_scene_wall_seconds=sum(r["total_wall_seconds"] for r in rows) / len(rows),
        throughput_seconds_per_scene=elapsed / len(rows),
        scenes=rows,
    )
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
