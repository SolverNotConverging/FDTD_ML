#!/usr/bin/env python3
"""Restartable low-budget PEC-cylinder mesh-candidate search."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scattermesh import (
    PEC,
    Circle,
    Grid,
    PlaneWave,
    circular_interface_axes,
    density_axis,
    simulate_cuda,
)
from scattermesh.analytic import cylinder_far_field
from scattermesh.metrics import scattering_loss

DOMAIN = 1.2
FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
CELLS = (32, 48, 64, 96)
SCENES = (
    dict(scene_id="small_axis", radius=0.045, center=(0.5573, 0.6241), angle=0.0),
    dict(scene_id="small_oblique", radius=0.045, center=(0.6387, 0.5669), angle=1.17),
    dict(scene_id="medium_oblique", radius=0.080, center=(0.6130, 0.5910), angle=0.70),
    dict(scene_id="large_reverse", radius=0.120, center=(0.6257, 0.5483), angle=3.91),
)
CANDIDATES = (
    dict(candidate="uniform", kind="uniform", max_ratio=1.0),
    dict(
        candidate="interface_wide", kind="interface", width_factor=1.0, weight=0.75, max_ratio=1.4
    ),
    dict(
        candidate="interface_moderate",
        kind="interface",
        width_factor=0.75,
        weight=1.5,
        max_ratio=2.0,
    ),
    dict(
        candidate="interface_medium", kind="interface", width_factor=0.5, weight=1.9, max_ratio=2.0
    ),
    dict(
        candidate="interface_strong", kind="interface", width_factor=0.35, weight=3.0, max_ratio=3.0
    ),
    dict(
        candidate="interface_narrow", kind="interface", width_factor=0.25, weight=3.0, max_ratio=3.0
    ),
    dict(candidate="region_wide", kind="region", width_factor=3.0, weight=0.5, max_ratio=1.4),
    dict(candidate="region_medium", kind="region", width_factor=2.0, weight=1.0, max_ratio=2.0),
    dict(candidate="region_strong", kind="region", width_factor=1.25, weight=1.5, max_ratio=2.0),
    dict(
        candidate="hybrid_wide",
        kind="hybrid",
        width_factor=1.0,
        weight=0.5,
        region_width_factor=2.5,
        region_weight=0.35,
        max_ratio=1.4,
    ),
    dict(
        candidate="hybrid_medium",
        kind="hybrid",
        width_factor=0.75,
        weight=0.8,
        region_width_factor=1.75,
        region_weight=0.6,
        max_ratio=2.0,
    ),
    dict(candidate="random_density_0", kind="random", random_index=0, max_ratio=2.0),
    dict(candidate="random_density_1", kind="random", random_index=1, max_ratio=2.0),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/mesh_headroom_expanded"))
    return parser.parse_args()


def sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "src/scattermesh").glob("*.py")), Path(__file__).resolve()]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def definitions():
    rows = []
    for scene in SCENES:
        for cells in CELLS:
            for candidate in CANDIDATES:
                config = dict(
                    schema_version=1,
                    scene=scene,
                    cells=cells,
                    **candidate,
                    duration=50e-9,
                    pml_thickness=0.15,
                    pec_mode="enlarged",
                    frequencies=FREQUENCIES.tolist(),
                    angles=len(ANGLES),
                )
                if config["kind"] == "random":
                    token = f"{scene['scene_id']}:{cells}:{config['random_index']}"
                    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
                    rng = np.random.default_rng(seed)
                    config["random_seed"] = seed
                    config["random_foci_x"] = [
                        [
                            float(rng.uniform(0.2, 1.0)),
                            float(rng.uniform(0.06, 0.20)),
                            float(rng.uniform(0.2, 0.9)),
                        ]
                        for _ in range(3)
                    ]
                    config["random_foci_y"] = [
                        [
                            float(rng.uniform(0.2, 1.0)),
                            float(rng.uniform(0.06, 0.20)),
                            float(rng.uniform(0.2, 0.9)),
                        ]
                        for _ in range(3)
                    ]
                config["case_id"] = f"{scene['scene_id']}_n{cells}_{candidate['candidate']}"
                rows.append(config)
    return rows


def make_grid(config):
    kind = config["kind"]
    if kind == "uniform":
        x = y = np.linspace(0, DOMAIN, config["cells"] + 1)
    elif kind == "interface":
        x, y = circular_interface_axes(
            DOMAIN,
            config["cells"],
            config["scene"]["center"],
            config["scene"]["radius"],
            width=config["width_factor"] * config["scene"]["radius"],
            weight=config["weight"],
        )
    elif kind in {"region", "hybrid"}:
        center, radius = config["scene"]["center"], config["scene"]["radius"]
        x_foci = [(center[0], config["width_factor"] * radius, config["weight"])]
        y_foci = [(center[1], config["width_factor"] * radius, config["weight"])]
        if kind == "hybrid":
            for offset in (-radius, radius):
                x_foci.append(
                    (center[0] + offset, config["width_factor"] * radius, config["weight"])
                )
                y_foci.append(
                    (center[1] + offset, config["width_factor"] * radius, config["weight"])
                )
            x_foci[0] = (
                center[0],
                config["region_width_factor"] * radius,
                config["region_weight"],
            )
            y_foci[0] = (
                center[1],
                config["region_width_factor"] * radius,
                config["region_weight"],
            )
        x = density_axis(DOMAIN, config["cells"], x_foci)
        y = density_axis(DOMAIN, config["cells"], y_foci)
    elif kind == "random":
        x = density_axis(DOMAIN, config["cells"], config["random_foci_x"])
        y = density_axis(DOMAIN, config["cells"], config["random_foci_y"])
    else:
        raise ValueError(f"Unsupported candidate kind: {kind}")
    return Grid(x, y, max_ratio=config["max_ratio"])


def analytic_reference(scene, source):
    return np.array(
        [
            cylinder_far_field(
                scene["radius"],
                PEC(),
                frequency,
                ANGLES,
                scene["angle"],
                center=scene["center"],
                incident_origin=source.origin,
            )
            for frequency in FREQUENCIES
        ]
    )


def valid_cache(record_path, arrays_path, fingerprint):
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        if record.get("fingerprint") == fingerprint and record.get("status") == "mesh_infeasible":
            return record
        if not arrays_path.exists():
            return None
        with np.load(arrays_path) as arrays:
            valid = (
                record.get("fingerprint") == fingerprint
                and record.get("backend") == "torch_cuda"
                and arrays["complex_far_field"].shape == (len(FREQUENCIES), len(ANGLES))
                and np.isfinite(arrays["complex_far_field"]).all()
                and np.isfinite(arrays["analytic_complex_far_field"]).all()
            )
        return record if valid else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def run_case(config, output, device, sources):
    fingerprint = sha256_json(dict(config=config, sources=sources))
    directory = output / "cases" / config["case_id"]
    record_path, arrays_path = directory / "record.json", directory / "spectra.npz"
    cached = valid_cache(record_path, arrays_path, fingerprint)
    if cached is not None:
        return cached, True
    directory.mkdir(parents=True, exist_ok=True)
    grid = make_grid(config)
    scene = config["scene"]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=scene["angle"], origin=(0.6, 0.6))
    try:
        result = simulate_cuda(
            grid,
            [Circle(scene["center"], scene["radius"], PEC())],
            source,
            frequencies=FREQUENCIES,
            duration=config["duration"],
            pml_thickness=config["pml_thickness"],
            pec_mode=config["pec_mode"],
            device=device,
            dtype="float64",
        )
    except ValueError as error:
        reason = str(error)
        if "PEC splits an edge" not in reason and "PEC geometry is unresolved" not in reason:
            raise
        record = dict(
            schema_version=1,
            case_id=config["case_id"],
            fingerprint=fingerprint,
            config=config,
            accepted=False,
            status="mesh_infeasible",
            reason=reason,
            requested_device=device,
        )
        atomic_json(record_path, record)
        return record, False
    field = result.monitor.normalized_far_field(ANGLES)
    reference = analytic_reference(scene, source)
    loss = scattering_loss(field, reference)
    record = dict(
        schema_version=1,
        case_id=config["case_id"],
        fingerprint=fingerprint,
        config=config,
        accepted=bool(
            np.isfinite(field).all() and result.diagnostics["tail_peak_over_global_peak"] < 1e-5
        ),
        status=(
            "accepted" if result.diagnostics["tail_peak_over_global_peak"] < 1e-5 else "unsettled"
        ),
        **loss,
        **result.diagnostics,
    )
    atomic_npz(
        arrays_path,
        x=grid.x,
        y=grid.y,
        frequencies=FREQUENCIES,
        angles=ANGLES,
        complex_far_field=field,
        analytic_complex_far_field=reference,
        scattering_width=2 * np.pi * abs(field) ** 2,
        analytic_scattering_width=2 * np.pi * abs(reference) ** 2,
    )
    atomic_json(record_path, record)
    return record, False


def pareto(records):
    ordered = sorted(records, key=lambda row: (row["cell_updates"], row["joint_scattering_loss"]))
    frontier, best_loss = [], np.inf
    for row in ordered:
        if row["joint_scattering_loss"] < best_loss:
            frontier.append(row["case_id"])
            best_loss = row["joint_scattering_loss"]
    return frontier


def summarize(output, sources):
    expected = definitions()
    records = []
    for config in expected:
        directory = output / "cases" / config["case_id"]
        fingerprint = sha256_json(dict(config=config, sources=sources))
        record = valid_cache(directory / "record.json", directory / "spectra.npz", fingerprint)
        if record is None:
            raise ValueError(f"Missing or stale pilot case: {config['case_id']}")
        records.append(record)
    scenes, all_accepted = {}, [row for row in records if row["accepted"]]
    for scene in SCENES:
        scene_id = scene["scene_id"]
        selected = [row for row in all_accepted if row["config"]["scene"]["scene_id"] == scene_id]
        budgets = {}
        for cells in CELLS:
            uniform = next(
                (
                    row
                    for row in selected
                    if row["config"]["cells"] == cells and row["config"]["candidate"] == "uniform"
                ),
                None,
            )
            if uniform is None:
                budgets[str(cells)] = dict(status="uniform_baseline_not_accepted")
                continue
            feasible = [row for row in selected if row["cell_updates"] <= uniform["cell_updates"]]
            best = min(feasible, key=lambda row: row["joint_scattering_loss"])
            same_cells = [row for row in selected if row["config"]["cells"] == cells]
            best_same = min(same_cells, key=lambda row: row["joint_scattering_loss"])
            budgets[str(cells)] = dict(
                uniform_case=uniform["case_id"],
                uniform_updates=uniform["cell_updates"],
                uniform_loss=uniform["joint_scattering_loss"],
                best_equal_or_lower_updates=best["case_id"],
                best_loss=best["joint_scattering_loss"],
                improvement=uniform["joint_scattering_loss"] / best["joint_scattering_loss"],
                best_same_cells=best_same["case_id"],
                best_same_cells_loss=best_same["joint_scattering_loss"],
                best_same_cells_update_ratio=best_same["cell_updates"] / uniform["cell_updates"],
            )
        scenes[scene_id] = dict(budgets=budgets, pareto_cases=pareto(selected))
    infeasible = [row for row in records if row["status"] == "mesh_infeasible"]
    unsettled = [row for row in records if row["status"] == "unsettled"]
    summary = dict(
        schema_version=1,
        decision="complete" if not unsettled else "has_unsettled_cases",
        case_count=len(records),
        accepted_count=len(all_accepted),
        mesh_infeasible_count=len(infeasible),
        unsettled_count=len(unsettled),
        loss="complex far-field + floored log scattering width",
        cost="Nx*Ny*Nt",
        scenes=scenes,
    )
    atomic_json(output / "report.json", summary)
    plot(records, output / "headroom.png")
    return summary


def plot(records, path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, scene in zip(axes.flat, SCENES):
        selected = [
            row
            for row in records
            if row["accepted"] and row["config"]["scene"]["scene_id"] == scene["scene_id"]
        ]
        uniform = [row for row in selected if row["config"]["candidate"] == "uniform"]
        focused = [row for row in selected if row["config"]["candidate"] != "uniform"]
        axis.scatter(
            [row["cell_updates"] for row in focused],
            [row["joint_scattering_loss"] for row in focused],
            s=22,
            alpha=0.65,
            label="nonuniform candidates",
        )
        axis.plot(
            [row["cell_updates"] for row in uniform],
            [row["joint_scattering_loss"] for row in uniform],
            "o-",
            label="uniform",
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
    sources = source_hashes()
    if args.summarize:
        report = summarize(args.output, sources)
        print(json.dumps(report, indent=2))
        return
    cases = definitions()
    selected = [case for index, case in enumerate(cases) if index % args.shards == args.shard]
    for index, config in enumerate(selected, 1):
        record, cached = run_case(config, args.output, args.device, sources)
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
