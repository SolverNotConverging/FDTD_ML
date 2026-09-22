#!/usr/bin/env python3
"""Evaluate learned and uniform meshes on the frozen circle OOD suite."""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scattermesh import Circle, Grid, Material, PlaneWave, simulate_cuda
from scattermesh.analytic import cylinder_far_field
from scattermesh.distillation import conditioning_features, probability_axis, rasterize_circle
from scattermesh.metrics import scattering_loss
from scattermesh.model import AxisDensityUNet

ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def source_hashes(checkpoint, plan):
    root = Path(__file__).resolve().parents[1]
    result = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }
    result.update(
        checkpoint=sha256_file(checkpoint),
        plan=sha256_file(plan),
        runner=sha256_file(__file__),
    )
    return result


def stable_shard(sample_id, shards):
    digest = hashlib.sha256(sample_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shards


def mesh_cases(plan, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    max_ratio = float(plan["max_grading_ratio"])
    cases = []
    with torch.no_grad():
        for example in plan["examples"]:
            raster = torch.from_numpy(rasterize_circle(example, 128))[None]
            features = torch.from_numpy(conditioning_features(example, max_ratio=max_ratio))[None]
            profiles = model(raster, features)[0].numpy()
            learned_x, x_repair = probability_axis(
                profiles[0], example["cells_x"], max_ratio=max_ratio
            )
            learned_y, y_repair = probability_axis(
                profiles[1], example["cells_y"], max_ratio=max_ratio
            )
            uniform_x = np.linspace(0, 1.2, example["cells_x"] + 1)
            uniform_y = np.linspace(0, 1.2, example["cells_y"] + 1)
            for mesh_kind, x, y, repairs in (
                ("uniform", uniform_x, uniform_y, (0.0, 0.0)),
                ("cnn", learned_x, learned_y, (x_repair, y_repair)),
            ):
                cases.append(
                    {
                        "case_id": f"{example['sample_id']}_{mesh_kind}",
                        "sample_id": example["sample_id"],
                        "mesh_kind": mesh_kind,
                        "example": example,
                        "x": x,
                        "y": y,
                        "axis_hash": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                        "max_grading_ratio": max_ratio,
                        "x_uniform_repair_fraction": repairs[0],
                        "y_uniform_repair_fraction": repairs[1],
                    }
                )
    return cases


def valid_attempt(directory, fingerprint):
    record_path, arrays_path = directory / "record.json", directory / "spectra.npz"
    if not record_path.exists() or not arrays_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        with np.load(arrays_path) as arrays:
            finite = np.isfinite(arrays["complex_far_field"]).all()
        return record if record.get("fingerprint") == fingerprint and finite else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def run_attempt(case, duration, output, device, sources, evaluation_id):
    example = case["example"]
    config = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "case_id": case["case_id"],
        "sample_id": case["sample_id"],
        "mesh_kind": case["mesh_kind"],
        "duration_s": duration,
        "axis_hash": case["axis_hash"],
        "max_grading_ratio": case["max_grading_ratio"],
        "x_uniform_repair_fraction": case["x_uniform_repair_fraction"],
        "y_uniform_repair_fraction": case["y_uniform_repair_fraction"],
        "example": example,
    }
    fingerprint = sha256_json({"config": config, "sources": sources})
    duration_ns = round(duration * 1e9)
    directory = output / "attempts" / case["case_id"] / f"duration_{duration_ns:03d}ns"
    cached = valid_attempt(directory, fingerprint)
    if cached is not None:
        return cached, directory, True
    grid = Grid(case["x"], case["y"], max_ratio=case["max_grading_ratio"])
    material = Material(example["epsilon_r"], example["sigma_e_s_per_m"])
    source = PlaneWave(
        1e9,
        1e-9,
        9e-9,
        angle=example["incidence_angle_rad"],
        origin=(0.6, 0.6),
    )
    frequencies = np.asarray(example["frequencies_hz"])
    result = simulate_cuda(
        grid,
        [Circle(example["center_m"], example["radius_m"], material)],
        source,
        frequencies=frequencies,
        duration=duration,
        pml_thickness=0.15,
        pec_mode=None,
        device=device,
        dtype="float64",
    )
    field = result.monitor.normalized_far_field(ANGLES)
    reference = np.array(
        [
            cylinder_far_field(
                example["radius_m"],
                material,
                frequency,
                ANGLES,
                example["incidence_angle_rad"],
                center=example["center_m"],
                incident_origin=source.origin,
            )
            for frequency in frequencies
        ]
    )
    accepted = bool(
        np.isfinite(field).all()
        and result.diagnostics["tail_peak_over_global_peak"] < 1e-5
    )
    record = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "fingerprint": fingerprint,
        "config": config,
        "accepted": accepted,
        "status": "accepted" if accepted else "unsettled",
        **scattering_loss(field, reference),
        **result.diagnostics,
    }
    atomic_npz(
        directory / "spectra.npz",
        x=case["x"],
        y=case["y"],
        frequencies=frequencies,
        angles=ANGLES,
        complex_far_field=field,
        analytic_complex_far_field=reference,
        scattering_width=2 * np.pi * abs(field) ** 2,
        analytic_scattering_width=2 * np.pi * abs(reference) ** 2,
    )
    atomic_json(directory / "record.json", record)
    return record, directory, False


def run_case(case, plan, output, device, sources):
    final = output / "cases" / case["case_id"]
    durations = plan["duration_schedule_s"]
    for duration in durations:
        record, attempt, cached = run_attempt(
            case, duration, output, device, sources, plan["evaluation_id"]
        )
        if record["accepted"] or duration == durations[-1]:
            final.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attempt / "record.json", final / "record.json")
            shutil.copy2(attempt / "spectra.npz", final / "spectra.npz")
            return record, cached
    raise RuntimeError("Unreachable duration schedule")


def aggregate(rows):
    improvements = [row["improvement_over_uniform"] for row in rows]
    complex_improvements = [row["complex_improvement_over_uniform"] for row in rows]
    rcs_improvements = [row["rcs_improvement_over_uniform"] for row in rows]
    return {
        "case_count": len(rows),
        "settled_pair_count": sum(row["settled_pair"] for row in rows),
        "meaningful_win_count": sum(value >= 1.05 for value in improvements),
        "meaningful_win_fraction": sum(value >= 1.05 for value in improvements)
        / len(rows),
        "minimum_improvement_over_uniform": min(improvements),
        "median_improvement_over_uniform": float(np.median(improvements)),
        "median_complex_improvement_over_uniform": float(np.median(complex_improvements)),
        "median_rcs_improvement_over_uniform": float(np.median(rcs_improvements)),
        "maximum_nt_ratio_to_uniform": max(row["nt_ratio_to_uniform"] for row in rows),
    }


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: aggregate(selected) for name, selected in sorted(groups.items())}


def summarize(plan, cases, output):
    exponent = float(plan["ranking"]["nt_cost_exponent"])
    by_sample = defaultdict(dict)
    examples = {}
    for case in cases:
        path = output / "cases" / case["case_id"] / "record.json"
        if not path.exists():
            raise ValueError(f"Missing generalization case: {case['case_id']}")
        by_sample[case["sample_id"]][case["mesh_kind"]] = json.loads(path.read_text())
        examples[case["sample_id"]] = case["example"]
    rows = []
    for sample_id, records in sorted(by_sample.items()):
        uniform, learned = records["uniform"], records["cnn"]
        example = examples[sample_id]
        uniform_score = uniform["joint_scattering_loss"]
        learned_score = learned["joint_scattering_loss"] * (
            learned["Nt"] / uniform["Nt"]
        ) ** exponent
        rows.append(
            {
                "sample_id": sample_id,
                "size_regime": example["size_regime"],
                "radius_m": example["radius_m"],
                "position_id": example["position_id"],
                "center_m": example["center_m"],
                "material_id": example["material_id"],
                "epsilon_r": example["epsilon_r"],
                "incidence_angle_rad": example["incidence_angle_rad"],
                "budget": example["cells_x"],
                "settled_pair": bool(uniform["accepted"] and learned["accepted"]),
                "uniform_score": uniform_score,
                "learned_score": learned_score,
                "improvement_over_uniform": uniform_score / learned_score,
                "complex_improvement_over_uniform": uniform["complex_mse_loss"]
                / max(learned["complex_mse_loss"], np.finfo(np.float64).tiny),
                "rcs_improvement_over_uniform": uniform["rcs_log_loss"]
                / max(learned["rcs_log_loss"], np.finfo(np.float64).tiny),
                "nt_ratio_to_uniform": learned["Nt"] / uniform["Nt"],
                "x_uniform_repair_fraction": learned["config"][
                    "x_uniform_repair_fraction"
                ],
                "y_uniform_repair_fraction": learned["config"][
                    "y_uniform_repair_fraction"
                ],
            }
        )
    overall = aggregate(rows)
    by_size = grouped(rows, lambda row: row["size_regime"])
    by_position = grouped(rows, lambda row: row["position_id"])
    gate = plan["gate"]
    checks = {
        "all_cases_settled": (
            not gate["require_all_cases_settled"]
            or overall["settled_pair_count"] == overall["case_count"]
        ),
        "overall_meaningful_win_fraction": overall["meaningful_win_fraction"]
        >= gate["minimum_meaningful_win_fraction"],
        "each_size_regime_win_fraction": all(
            value["meaningful_win_fraction"]
            >= gate["minimum_each_size_regime_win_fraction"]
            for value in by_size.values()
        ),
        "each_position_median_improvement": all(
            value["median_improvement_over_uniform"]
            >= gate["minimum_each_position_median_improvement"]
            for value in by_position.values()
        ),
        "overall_median_improvement": overall["median_improvement_over_uniform"]
        >= gate["minimum_overall_median_improvement"],
        "worst_case_improvement": overall["minimum_improvement_over_uniform"]
        >= gate["minimum_worst_case_improvement"],
    }
    report = {
        "schema_version": 1,
        "evaluation_id": plan["evaluation_id"],
        "decision": (
            "passes_circle_position_scale_generalization"
            if all(checks.values())
            else "fails_circle_position_scale_generalization"
        ),
        "checks": checks,
        "gate": gate,
        "ranking": plan["ranking"],
        "overall": overall,
        "by_size_regime": by_size,
        "by_position": by_position,
        "by_material": grouped(rows, lambda row: row["material_id"]),
        "by_budget": grouped(rows, lambda row: row["budget"]),
        "cases": rows,
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/circle_position_scale_generalization.json")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    plan = json.loads(args.plan.read_text())
    cases = mesh_cases(plan, args.checkpoint)
    if args.summarize:
        summarize(plan, cases, args.output)
        return
    sources = source_hashes(args.checkpoint, args.plan)
    sample_ids = {
        example["sample_id"]
        for example in plan["examples"]
        if stable_shard(example["sample_id"], args.shards) == args.shard
    }
    selected = [case for case in cases if case["sample_id"] in sample_ids]
    for index, case in enumerate(selected, 1):
        record, cached = run_case(case, plan, args.output, args.device, sources)
        print(
            f"[{index}/{len(selected)}] {case['case_id']} status={record['status']} "
            f"loss={record['joint_scattering_loss']:.6g} "
            f"{'cached' if cached else 'ran'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
