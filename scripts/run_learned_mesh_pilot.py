#!/usr/bin/env python3
"""Evaluate learned exact-budget meshes with the CUDA scattering solver."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scattermesh import Circle, Grid, Material, PlaneWave, simulate_cuda
from scattermesh.analytic import cylinder_far_field
from scattermesh.distillation import probability_axis
from scattermesh.metrics import scattering_loss
from scattermesh.model import AxisDensityUNet, MeshDistillationDataset

ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
DURATIONS = (70e-9, 140e-9, 560e-9)


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


def source_hashes(checkpoint, dataset):
    root = Path(__file__).resolve().parents[1]
    result = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }
    result["checkpoint"] = sha256_file(checkpoint)
    result["dataset"] = sha256_file(dataset)
    result[str(Path(__file__).resolve().relative_to(root))] = sha256_file(__file__)
    return result


def stable_shard(case_id, shards):
    """Assign cases reproducibly without coupling work to dataset ordering."""
    digest = hashlib.sha256(case_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shards


def predicted_cases(dataset_path, checkpoint_path, splits, max_ratio):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = AxisDensityUNet(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cases = []
    with torch.no_grad():
        for split in splits:
            dataset = MeshDistillationDataset(dataset_path, split)
            for index, example in enumerate(dataset.examples):
                batch = dataset[index]
                prediction = model(batch["raster"][None], batch["conditioning"][None])[0].numpy()
                x, x_repair = probability_axis(
                    prediction[0], example["cells_x"], max_ratio=max_ratio
                )
                y, y_repair = probability_axis(
                    prediction[1], example["cells_y"], max_ratio=max_ratio
                )
                axis_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
                cases.append(
                    {
                        "case_id": f"{example['sample_id']}_cnn",
                        "sample_id": example["sample_id"],
                        "split": split,
                        "example": example,
                        "x": x,
                        "y": y,
                        "axis_hash": axis_hash,
                        "max_grading_ratio": max_ratio,
                        "x_uniform_repair_fraction": x_repair,
                        "y_uniform_repair_fraction": y_repair,
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


def run_attempt(case, duration, output, device, sources):
    example = case["example"]
    config = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "sample_id": case["sample_id"],
        "split": case["split"],
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
        np.isfinite(field).all() and result.diagnostics["tail_peak_over_global_peak"] < 1e-5
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


def run_case(case, output, device, sources):
    final = output / "cases" / case["case_id"]
    for duration in DURATIONS:
        record, attempt, cached = run_attempt(case, duration, output, device, sources)
        if record["accepted"] or duration == DURATIONS[-1]:
            final.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attempt / "record.json", final / "record.json")
            shutil.copy2(attempt / "spectra.npz", final / "spectra.npz")
            return record, cached
    raise RuntimeError("Unreachable duration schedule")


def summarize(
    cases,
    output,
    candidate_output,
    exponent,
    success_decision="passes_physics_pilot",
):
    rows = []
    for case in cases:
        learned_path = output / "cases" / case["case_id"] / "record.json"
        if not learned_path.exists():
            raise ValueError(f"Missing learned case: {case['case_id']}")
        learned = json.loads(learned_path.read_text())
        example = case["example"]
        uniform = json.loads(
            (candidate_output / "cases" / f"{case['sample_id']}_uniform" / "record.json").read_text()
        )
        best = json.loads(
            (
                candidate_output
                / "cases"
                / f"{case['sample_id']}_{example['best_candidate']}"
                / "record.json"
            ).read_text()
        )
        learned_score = learned["joint_scattering_loss"] * (
            learned["Nt"] / uniform["Nt"]
        ) ** exponent
        uniform_score = uniform["joint_scattering_loss"]
        best_score = best["joint_scattering_loss"] * (best["Nt"] / uniform["Nt"]) ** exponent
        rows.append(
            {
                "case_id": case["case_id"],
                "sample_id": case["sample_id"],
                "split": case["split"],
                "budget": example["cells_x"],
                "accepted": learned["accepted"],
                "learned_loss": learned["joint_scattering_loss"],
                "learned_score": learned_score,
                "uniform_score": uniform_score,
                "best_candidate": example["best_candidate"],
                "best_teacher_score": best_score,
                "improvement_over_uniform": uniform_score / learned_score,
                "score_ratio_to_teacher": learned_score / best_score,
                "Nt_ratio_to_uniform": learned["Nt"] / uniform["Nt"],
            }
        )
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    splits = {}
    for split, selected in sorted(by_split.items()):
        improvements = [row["improvement_over_uniform"] for row in selected]
        teacher_ratios = [row["score_ratio_to_teacher"] for row in selected]
        splits[split] = {
            "case_count": len(selected),
            "accepted_count": sum(row["accepted"] for row in selected),
            "meaningful_uniform_wins": sum(value >= 1.05 for value in improvements),
            "p10_improvement_over_uniform": float(np.quantile(improvements, 0.10)),
            "median_improvement_over_uniform": float(np.median(improvements)),
            "minimum_improvement_over_uniform": min(improvements),
            "median_score_ratio_to_teacher": float(np.median(teacher_ratios)),
            "p90_score_ratio_to_teacher": float(np.quantile(teacher_ratios, 0.90)),
            "maximum_score_ratio_to_teacher": max(teacher_ratios),
            "maximum_learned_loss": max(row["learned_loss"] for row in selected),
            "maximum_nt_ratio_to_uniform": max(row["Nt_ratio_to_uniform"] for row in selected),
        }
    checks = {
        "all_cases_settled": all(row["accepted"] for row in rows),
        "each_split_has_75_percent_meaningful_wins": all(
            value["meaningful_uniform_wins"] >= 0.75 * value["case_count"]
            for value in splits.values()
        ),
        "each_split_median_improvement_exceeds_1_2": all(
            value["median_improvement_over_uniform"] > 1.2 for value in splits.values()
        ),
        "each_split_median_teacher_gap_below_1_5": all(
            value["median_score_ratio_to_teacher"] < 1.5 for value in splits.values()
        ),
    }
    report = {
        "schema_version": 1,
        "decision": (
            success_decision
            if all(checks.values())
            else success_decision.replace("passes_", "fails_", 1)
        ),
        "checks": checks,
        "case_count": len(rows),
        "status_counts": dict(sorted(Counter("accepted" if r["accepted"] else "unsettled" for r in rows).items())),
        "ranking": {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": exponent},
        "splits": splits,
        "cases": rows,
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-mode", choices=("hash", "index"), default="hash")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--success-decision",
        default="passes_physics_pilot",
        choices=("passes_physics_pilot", "passes_frozen_physics_evaluation"),
    )
    parser.add_argument("--max-ratio", type=float, default=3.0)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    cases = predicted_cases(
        args.dataset, args.checkpoint, ("validation", "test"), args.max_ratio
    )
    metadata = json.loads(args.dataset.read_text())
    exponent = float(metadata["ranking"]["nt_cost_exponent"])
    if args.summarize:
        summarize(
            cases,
            args.output,
            args.candidate_output,
            exponent,
            success_decision=args.success_decision,
        )
        return
    sources = source_hashes(args.checkpoint, args.dataset)
    if args.shard_mode == "hash":
        selected = [
            case for case in cases if stable_shard(case["case_id"], args.shards) == args.shard
        ]
    else:
        selected = [
            case for index, case in enumerate(cases) if index % args.shards == args.shard
        ]
    for index, case in enumerate(selected, 1):
        record, cached = run_case(case, args.output, args.device, sources)
        print(
            f"[{index}/{len(selected)}] {case['case_id']} status={record['status']} "
            f"loss={record['joint_scattering_loss']:.6g} "
            f"{'cached' if cached else 'ran'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
