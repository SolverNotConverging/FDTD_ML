"""Start/resume the v6 campaign in a detached Linux session; logs stay in its output."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference_v6"))
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    toolkit = Path("/usr/local/cuda-12.6/lib64")
    if toolkit.is_dir():
        env["LD_LIBRARY_PATH"] = str(toolkit) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        "-u",
        "-m",
        "fdtdmesh.data.campaign",
        "--output",
        str(output),
        *extra,
    ]
    with (output / "coordinator.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (output / "launch.json").write_text(
        json.dumps(dict(pid=process.pid, command=command, cwd=str(root)), indent=2)
    )
    print(f"Coordinator PID {process.pid}; output {output}", flush=True)


if __name__ == "__main__":
    main()
