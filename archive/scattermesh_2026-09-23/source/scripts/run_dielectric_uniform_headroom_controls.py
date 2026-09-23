#!/usr/bin/env python3
"""Dense uniform controls for the simple dielectric mesh-headroom search."""

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT = load("scattermesh_headroom_pilot", ROOT / "scripts/pilot_mesh_headroom.py")
DIELECTRIC = load(
    "scattermesh_dielectric_headroom", ROOT / "scripts/pilot_dielectric_mesh_headroom.py"
)
CONTROLS = load("scattermesh_uniform_controls", ROOT / "scripts/run_uniform_headroom_controls.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/dielectric_mesh_headroom"))
    return parser.parse_args()


def definitions():
    return CONTROLS.definitions(DIELECTRIC.SCENES, duration=DIELECTRIC.DURATION, pec_mode=None)


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    cases = definitions()
    if args.summarize:
        CONTROLS.summarize(
            args.output,
            DIELECTRIC.SCENES,
            DIELECTRIC.definitions(),
            cases,
            DIELECTRIC.CELLS,
        )
        return
    sources = PILOT.source_hashes()
    selected = [case for index, case in enumerate(cases) if index % args.shards == args.shard]
    for index, config in enumerate(selected, 1):
        record, cached = PILOT.run_case(config, args.output, args.device, sources)
        if record["status"] == "mesh_infeasible":
            outcome = f"infeasible={record['reason']}"
        else:
            outcome = f"loss={record['joint_scattering_loss']:.6g} updates={record['cell_updates']}"
        print(
            f"[{index}/{len(selected)}] {config['case_id']} {outcome} "
            f"{'cached' if cached else 'ran'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
