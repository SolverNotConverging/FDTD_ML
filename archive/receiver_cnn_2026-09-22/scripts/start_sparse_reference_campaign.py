"""Generate the sparse supplement, then publish a combined corpus for later training."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fdtdmesh.data.supplement import merge_campaigns
from fdtdmesh.evaluation.references import _write_json

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_json(
            args.output / "workflow_stage.json", dict(stage="sparse_references", time=time.time())
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "fdtdmesh.data.campaign",
            "--output",
            str(args.output),
            "--generator-version",
            "7",
            "--train",
            "800",
            "--validation",
            "100",
            "--test",
            "100",
            "--gpus",
            *map(str, args.gpus),
            "--max-reference-level",
            "2048",
            "--no-extend-nonconverged",
            "--material-averaging",
            "sampled",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        _write_json(
            args.output / "workflow_stage.json",
            dict(stage="combining_references", time=time.time()),
        )
        manifest = merge_campaigns([args.base, args.output], args.combined_output)
        _write_json(
            args.output / "workflow_stage.json",
            dict(
                stage="ready_for_training",
                dataset_id=manifest["dataset_id"],
                scenes=len(manifest["scenes"]),
                combined_output=str(args.combined_output),
                time=time.time(),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reference_sparse_v7_1000")
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/reference_v6_2000")
    parser.add_argument(
        "--combined-output", type=Path, default=ROOT / "artifacts/reference_combined_v7_3000"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    for key in ("output", "base", "combined_output"):
        setattr(args, key, getattr(args, key).resolve())
    if not (args.base / "complete.json").exists():
        parser.error("The base reference campaign must be complete")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.foreground:
        try:
            run(args)
        except Exception as error:
            _write_json(
                args.output / "workflow_failure.json", dict(error=str(error), time=time.time())
            )
            raise
        return
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--foreground",
        "--output",
        str(args.output),
        "--base",
        str(args.base),
        "--combined-output",
        str(args.combined_output),
        "--gpus",
        *map(str, args.gpus),
    ]
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
        args.output / "workflow_launch.json",
        dict(pid=process.pid, command=command, time=time.time()),
    )
    print(json.dumps(dict(pid=process.pid, output=str(args.output))))


if __name__ == "__main__":
    main()
