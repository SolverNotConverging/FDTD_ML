#!/usr/bin/env python3
"""Restartable simple dielectric-cylinder mesh-headroom search."""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "scripts/pilot_mesh_headroom.py"
SPEC = importlib.util.spec_from_file_location("scattermesh_headroom_pilot", PILOT_PATH)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)

SCENES = (
    dict(
        scene_id="eps2_small_axis",
        radius=0.045,
        center=(0.5573, 0.6241),
        angle=0.0,
        material=dict(epsilon_r=2.0, sigma_e=0.0),
    ),
    dict(
        scene_id="eps2_medium_oblique",
        radius=0.080,
        center=(0.6387, 0.5669),
        angle=1.17,
        material=dict(epsilon_r=2.0, sigma_e=0.0),
    ),
    dict(
        scene_id="eps4_medium_oblique",
        radius=0.080,
        center=(0.6130, 0.5910),
        angle=0.70,
        material=dict(epsilon_r=4.0, sigma_e=0.0),
    ),
    dict(
        scene_id="eps4_lossy_reverse",
        radius=0.070,
        center=(0.6257, 0.5483),
        angle=3.91,
        material=dict(epsilon_r=4.0, sigma_e=0.05),
    ),
)
CELLS = (32, 48, 64, 96)
DURATION = 70e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/dielectric_mesh_headroom"))
    return parser.parse_args()


def definitions():
    return PILOT.definitions(
        SCENES,
        CELLS,
        PILOT.CANDIDATES,
        duration=DURATION,
        pec_mode=None,
    )


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    source_hashes = PILOT.source_hashes()
    cases = definitions()
    if args.summarize:
        report = PILOT.summarize(args.output, source_hashes, cases, SCENES, CELLS)
        print(json.dumps(report, indent=2))
        return
    selected = [case for index, case in enumerate(cases) if index % args.shards == args.shard]
    for index, config in enumerate(selected, 1):
        record, cached = PILOT.run_case(config, args.output, args.device, source_hashes)
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
