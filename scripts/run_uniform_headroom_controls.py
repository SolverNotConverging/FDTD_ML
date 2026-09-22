#!/usr/bin/env python3
"""Run dense uniform controls for the expanded mesh-headroom Pareto test."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "scripts/pilot_mesh_headroom.py"
SPEC = importlib.util.spec_from_file_location("scattermesh_headroom_pilot", PILOT_PATH)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)

CONTROL_CELLS = tuple(
    cells for cells in sorted({*range(32, 129, 4), 65, 66, 67}) if cells not in PILOT.CELLS
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/mesh_headroom_expanded"))
    return parser.parse_args()


def definitions():
    rows = []
    for scene in PILOT.SCENES:
        for cells in CONTROL_CELLS:
            rows.append(
                dict(
                    schema_version=1,
                    scene=scene,
                    cells=cells,
                    candidate="uniform",
                    kind="uniform",
                    max_ratio=1.0,
                    duration=50e-9,
                    pml_thickness=0.15,
                    pec_mode="enlarged",
                    frequencies=PILOT.FREQUENCIES.tolist(),
                    angles=len(PILOT.ANGLES),
                    case_id=f"{scene['scene_id']}_n{cells}_uniform",
                )
            )
    return rows


def sources():
    # Numerical identity is fully specified by the pilot solver/mesher sources and
    # each explicit uniform-grid config. Summary/reporting edits do not stale fields.
    return PILOT.source_hashes()


def load_verified(config, output, source_hashes):
    fingerprint = PILOT.sha256_json(dict(config=config, sources=source_hashes))
    directory = output / "cases" / config["case_id"]
    record = PILOT.valid_cache(directory / "record.json", directory / "spectra.npz", fingerprint)
    if record is None:
        raise ValueError(f"Missing or stale uniform control: {config['case_id']}")
    return record


def summarize(output):
    pilot_sources = PILOT.source_hashes()
    PILOT.summarize(output, pilot_sources)
    pilot_records = []
    for config in PILOT.definitions():
        fingerprint = PILOT.sha256_json(dict(config=config, sources=pilot_sources))
        directory = output / "cases" / config["case_id"]
        record = PILOT.valid_cache(
            directory / "record.json", directory / "spectra.npz", fingerprint
        )
        if record is None:
            raise ValueError(f"Missing or stale expanded case: {config['case_id']}")
        pilot_records.append(record)
    control_sources = sources()
    controls = [load_verified(config, output, control_sources) for config in definitions()]
    accepted = [record for record in [*pilot_records, *controls] if record["accepted"]]
    scenes = {}
    for scene in PILOT.SCENES:
        scene_id = scene["scene_id"]
        selected = [
            record for record in accepted if record["config"]["scene"]["scene_id"] == scene_id
        ]
        uniform = [record for record in selected if record["config"]["kind"] == "uniform"]
        nonuniform = [record for record in selected if record["config"]["kind"] != "uniform"]
        frontier_ids = PILOT.pareto(selected)
        nonuniform_frontier = [record for record in nonuniform if record["case_id"] in frontier_ids]
        comparisons = []
        for candidate in sorted(nonuniform_frontier, key=lambda row: row["cell_updates"]):
            affordable = [
                control
                for control in uniform
                if control["cell_updates"] <= candidate["cell_updates"]
            ]
            baseline = min(affordable, key=lambda row: row["joint_scattering_loss"])
            comparisons.append(
                dict(
                    candidate=candidate["case_id"],
                    candidate_updates=candidate["cell_updates"],
                    candidate_loss=candidate["joint_scattering_loss"],
                    uniform_control=baseline["case_id"],
                    uniform_updates=baseline["cell_updates"],
                    uniform_loss=baseline["joint_scattering_loss"],
                    nonuniform_advantage=baseline["joint_scattering_loss"]
                    / candidate["joint_scattering_loss"],
                )
            )
        scenes[scene_id] = dict(
            pareto_cases=frontier_ids,
            nonuniform_pareto_comparisons=comparisons,
        )
    advantages = [
        row["nonuniform_advantage"]
        for scene in scenes.values()
        for row in scene["nonuniform_pareto_comparisons"]
    ]
    report = dict(
        schema_version=1,
        decision="headroom_found" if advantages and max(advantages) > 1 else "no_headroom",
        pilot_case_count=len(pilot_records),
        uniform_control_count=len(controls),
        accepted_count=len(accepted),
        maximum_nonuniform_advantage=max(advantages, default=None),
        scenes=scenes,
    )
    PILOT.atomic_json(output / "dense_uniform_report.json", report)
    plot(accepted, output / "dense_uniform_headroom.png")
    print(json.dumps(report, indent=2))


def plot(records, path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, scene in zip(axes.flat, PILOT.SCENES):
        selected = [
            record
            for record in records
            if record["config"]["scene"]["scene_id"] == scene["scene_id"]
        ]
        uniform = sorted(
            (record for record in selected if record["config"]["kind"] == "uniform"),
            key=lambda record: record["cell_updates"],
        )
        nonuniform = [record for record in selected if record["config"]["kind"] != "uniform"]
        axis.scatter(
            [record["cell_updates"] for record in nonuniform],
            [record["joint_scattering_loss"] for record in nonuniform],
            s=20,
            alpha=0.55,
            label="nonuniform",
        )
        axis.plot(
            [record["cell_updates"] for record in uniform],
            [record["joint_scattering_loss"] for record in uniform],
            "o-",
            markersize=3,
            linewidth=1.2,
            label="uniform controls",
        )
        axis.set(xscale="log", yscale="log", title=scene["scene_id"])
        axis.grid(True, which="both", alpha=0.25)
    axes[1, 0].set_xlabel("cell updates (Nx Ny Nt)")
    axes[1, 1].set_xlabel("cell updates (Nx Ny Nt)")
    axes[0, 0].set_ylabel("joint scattering loss")
    axes[1, 0].set_ylabel("joint scattering loss")
    axes[0, 0].legend(fontsize=8)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    args = parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    if args.summarize:
        summarize(args.output)
        return
    source_hashes = sources()
    selected = [
        config for index, config in enumerate(definitions()) if index % args.shards == args.shard
    ]
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
