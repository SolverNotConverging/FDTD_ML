#!/usr/bin/env python3
"""Frozen sparse-pair physics evaluation for trained model-grid checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scattermesh import Grid, PlaneWave, simulate_cuda, sparse_scene_objects
from scattermesh.distillation import conditioning_features_v2, probability_axis, rasterize_scene
from scattermesh.generalization import widest_non_pml_monitor_bounds
from scattermesh.metrics import scattering_loss
from scattermesh.model import AxisDensityUNet

ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_pilots(pilots):
    result = {}
    for config_path, output in pilots:
        config_path, output = Path(config_path), Path(output)
        config = json.loads(config_path.read_text())
        report = json.loads((output / "report.json").read_text())
        if report["decision"] != "passes_sparse_pair_headroom":
            raise ValueError(f"Pilot has not qualified: {output}")
        for scene in config["scenes"]:
            if scene["scene_id"] in result:
                raise ValueError(f"Duplicate scene ID: {scene['scene_id']}")
            result[scene["scene_id"]] = (config, output, scene)
    return result


def _score_one(example, model, resolution, pilot, device, output, checkpoint_hash, sources):
    config, pilot_dir, scene = pilot
    cells = int(example["cells_x"])
    if cells != example["cells_y"]:
        raise ValueError("Pair physics evaluation currently uses square budgets")
    sample_id = example["sample_id"]
    record_path = output / "cases" / sample_id / "record.json"
    arrays_path = record_path.parent / "spectra.npz"
    baseline_case = f"candidate_{scene['scene_id']}_n{cells}_uniform"
    baseline_path = pilot_dir / "candidates" / baseline_case / "record.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline["accepted"]:
        raise ValueError(f"No accepted uniform baseline: {baseline_case}")
    fine_case = f"reference_{scene['scene_id']}_n{max(config['reference_budgets'])}"
    fine_dir = pilot_dir / "references" / fine_case
    fine_record = json.loads((fine_dir / "record.json").read_text())
    if not fine_record["accepted"]:
        raise ValueError(f"Fine reference is not settled: {fine_case}")
    fingerprint = _sha256_json(
        {
            "sample": example,
            "checkpoint": checkpoint_hash,
            "baseline": baseline["fingerprint"],
            "reference": fine_record["fingerprint"],
            "sources": sources,
            "protocol": {
                key: config[key]
                for key in (
                    "frequencies_hz",
                    "far_field_angle_count",
                    "duration_schedule_s",
                    "pml_thickness_m",
                    "tail_limit",
                    "material_samples_per_axis",
                )
            },
        }
    )
    if record_path.is_file() and arrays_path.is_file():
        cached = json.loads(record_path.read_text())
        if cached.get("fingerprint") == fingerprint:
            with np.load(arrays_path) as arrays:
                if arrays["complex_far_field"].shape == (
                    len(config["frequencies_hz"]),
                    config["far_field_angle_count"],
                ):
                    return cached

    raster = torch.from_numpy(rasterize_scene(example, resolution))[None]
    condition = torch.from_numpy(conditioning_features_v2(example, max_ratio=3.0))[None]
    with torch.no_grad():
        profiles = model(raster, condition)[0].cpu().numpy()
    x, x_repair = probability_axis(profiles[0], cells, max_ratio=3.0)
    y, y_repair = probability_axis(profiles[1], cells, max_ratio=3.0)
    grid = Grid(x, y, max_ratio=3.0)
    objects = sparse_scene_objects(scene)
    try:
        monitor = widest_non_pml_monitor_bounds(
            grid, [obj.bounds for obj in objects], config["pml_thickness_m"]
        )
    except ValueError as error:
        record = {
            "sample_id": sample_id,
            "fingerprint": fingerprint,
            "status": "mesh_infeasible",
            "reason": str(error),
            "split": example["split"],
            "shape_topology": example["shape_topology"],
            "material_topology": example["material_topology"],
            "gap_stratum": example["gap_stratum"],
            "cells": cells,
            "uniform_score": baseline["joint_scattering_loss"],
        }
        _atomic_npz(arrays_path, x=x, y=y, complex_far_field=np.zeros((len(config["frequencies_hz"]), config["far_field_angle_count"]), dtype=np.complex128))
        _atomic_json(record_path, record)
        return record
    source = PlaneWave(
        1e9, 1e-9, 9e-9,
        angle=scene["incidence_angle_rad"],
        origin=(config["domain_m"] / 2, config["domain_m"] / 2),
    )
    angles = np.linspace(0, 2 * np.pi, config["far_field_angle_count"], endpoint=False)
    with np.load(fine_dir / "spectra.npz") as arrays:
        reference = arrays["complex_far_field"].copy()
    attempts = []
    result = None
    field = None
    accepted = False
    for duration in config["duration_schedule_s"]:
        try:
            result = simulate_cuda(
                grid,
                objects,
                source,
                frequencies=np.asarray(config["frequencies_hz"]),
                duration=duration,
                pml_thickness=config["pml_thickness_m"],
                monitor_bounds=monitor,
                samples=config["material_samples_per_axis"],
                pec_mode=config.get("pec_mode", "conformal"),
                device=device,
                dtype="float64",
            )
        except ValueError as error:
            if (
                "PEC splits an edge" in str(error)
                or "PEC geometry is unresolved" in str(error)
                or ("Simulation requires" in str(error) and "exceeding" in str(error))
            ):
                record = {
                    "sample_id": sample_id,
                    "fingerprint": fingerprint,
                    "status": (
                        "hard_limit_unsettled"
                        if "Simulation requires" in str(error)
                        else "mesh_infeasible"
                    ),
                    "reason": str(error),
                    "split": example["split"],
                    "shape_topology": example["shape_topology"],
                    "material_topology": example["material_topology"],
                    "gap_stratum": example["gap_stratum"],
                    "cells": cells,
                    "uniform_score": baseline["joint_scattering_loss"],
                }
                _atomic_npz(arrays_path, x=x, y=y, complex_far_field=np.zeros_like(reference))
                _atomic_json(record_path, record)
                return record
            raise
        field = result.monitor.normalized_far_field(angles)
        tail = result.diagnostics["tail_peak_over_global_peak"]
        attempts.append({"duration_s": duration, "Nt": result.diagnostics["Nt"], "tail": tail})
        if np.isfinite(field).all() and tail < config["tail_limit"]:
            accepted = True
            break
    assert result is not None and field is not None
    losses = scattering_loss(field, reference)
    soft_score = losses["joint_scattering_loss"] * (
        result.diagnostics["Nt"] / baseline["Nt"]
    ) ** config["ranking"]["nt_cost_exponent"]
    record = {
        "schema_version": 1,
        "sample_id": sample_id,
        "fingerprint": fingerprint,
        "status": "accepted" if accepted else "hard_limit_unsettled",
        "split": example["split"],
        "shape_topology": example["shape_topology"],
        "material_topology": example["material_topology"],
        "gap_stratum": example["gap_stratum"],
        "cells": cells,
        "x_uniform_repair_fraction": x_repair,
        "y_uniform_repair_fraction": y_repair,
        "uniform_score": baseline["joint_scattering_loss"],
        "soft_nt_score": soft_score,
        "improvement_over_uniform": (
            baseline["joint_scattering_loss"] / soft_score if accepted else None
        ),
        "attempts": attempts,
        **losses,
        **result.diagnostics,
    }
    _atomic_npz(
        arrays_path,
        x=x,
        y=y,
        frequencies=np.asarray(config["frequencies_hz"]),
        angles=angles,
        complex_far_field=field,
        reference_complex_far_field=reference,
    )
    _atomic_json(record_path, record)
    return record


def _summarize(output, examples):
    records = []
    for example in examples:
        path = output / "cases" / example["sample_id"] / "record.json"
        if not path.is_file():
            raise ValueError(f"Missing case: {example['sample_id']}")
        records.append(json.loads(path.read_text()))
    split_reports = {}
    for split in ("validation", "test"):
        selected = [row for row in records if row["split"] == split]
        accepted = [row for row in selected if row["status"] == "accepted"]
        improvements = [row["improvement_over_uniform"] for row in accepted]
        strata = {}
        for field in ("shape_topology", "material_topology", "gap_stratum", "cells"):
            strata[field] = {}
            for name in sorted({str(row[field]) for row in selected}):
                subset = [row for row in selected if str(row[field]) == name]
                values = [row["improvement_over_uniform"] for row in subset if row["status"] == "accepted"]
                strata[field][name] = {
                    "case_count": len(subset),
                    "accepted_count": len(values),
                    "median_improvement": float(np.median(values)) if values else None,
                    "meaningful_win_fraction": sum(value >= 1.05 for value in values) / len(subset),
                }
        split_reports[split] = {
            "case_count": len(selected),
            "accepted_count": len(accepted),
            "meaningful_win_count": sum(value >= 1.05 for value in improvements),
            "meaningful_win_fraction": sum(value >= 1.05 for value in improvements) / len(selected),
            "median_improvement": float(np.median(improvements)) if improvements else None,
            "minimum_improvement": min(improvements) if improvements else None,
            "strata": strata,
        }
    report = {"schema_version": 1, "split_reports": split_reports, "cases": records}
    _atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pilot", nargs=2, action="append", metavar=("CONFIG", "OUTPUT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.dataset.read_text())
    examples = [
        item for item in metadata["examples"]
        if item["family"] == "sparse_pair" and item["split"] in {"validation", "test"}
    ]
    if args.summarize:
        print(json.dumps(_summarize(args.output, examples)["split_reports"], indent=2))
        return
    if args.checkpoint is None:
        raise ValueError("A checkpoint is required for physics runs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["config"].get("input_schema") != "sparse_v2":
        raise ValueError("Checkpoint does not use sparse-v2 input")
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    resolution = checkpoint["config"]["raster_resolution"]
    pilots = _load_pilots(args.pilot)
    sources = {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in sorted((ROOT / "src/scattermesh").glob("*.py"))
    }
    sources["runner"] = _sha256_file(__file__)
    checkpoint_hash = _sha256_file(args.checkpoint)
    for index, example in enumerate(examples):
        if index % args.shards != args.shard:
            continue
        record = _score_one(
            example, model, resolution, pilots[example["geometry_id"]],
            args.device, args.output, checkpoint_hash, sources,
        )
        print(
            f"[{index + 1}/{len(examples)}] {example['sample_id']} {record['status']} "
            f"improvement={record.get('improvement_over_uniform')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
