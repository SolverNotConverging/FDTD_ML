"""Launch the stratified v6 physics-error pilot as a detached process."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = [
    "validation-v6-000000",  # separated
    "validation-v6-000005",  # contact
    "validation-v6-000002",  # overlap
    "validation-v6-000007",  # nested
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/physics_pilot_v6"))
    parser.add_argument(
        "--references", type=Path, default=Path("artifacts/physics_pilot_v6/references")
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[48, 64, 96, 128])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--generate-missing-teachers", action="store_true")
    parser.add_argument("--meshing-time-limit", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    references = args.references if args.references.is_absolute() else ROOT / args.references
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "fdtdmesh.physics",
        "search",
        "--manifest",
        str(ROOT / "artifacts/reference_v6_2000/accepted_manifest.json"),
        "--teacher-targets",
        str(ROOT / "artifacts/training_v6_2000/teacher/targets.npz"),
        "--checkpoint",
        str(ROOT / "artifacts/training_v6_2000/pretraining/best.pt"),
        "--references",
        str(references),
        "--output",
        str(output / "search"),
        "--splits",
        "validation",
        "--scenes",
        *SCENES,
        "--budgets",
        *map(str, args.budgets),
        "--beta",
        "0.02",
        "--physics-weight",
        "0.5",
        "--perturbations",
        "2",
        "--meshing-time-limit",
        str(args.meshing_time_limit),
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
    if args.generate_missing_teachers:
        command.append("--generate-missing-teachers")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.6/lib64:" + env.get("LD_LIBRARY_PATH", "")
    with (output / "pilot.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {"pid": process.pid, "command": command, "time": time.time()}
    temporary = output / "launch.json.tmp"
    temporary.write_text(json.dumps(launch, indent=2), encoding="utf-8")
    temporary.replace(output / "launch.json")
    print(f"Physics pilot PID {process.pid}; output {output}")


if __name__ == "__main__":
    main()
