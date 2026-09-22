#!/usr/bin/env python3
"""Summarize float32 accuracy and throughput qualification artifacts."""

import json
import os
from pathlib import Path

import numpy as np

CASES = ("eps12_small", "eps30_lossy", "pec_enlarged")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_case(root, name):
    directory = root / name
    record = json.loads((directory / "record.json").read_text())
    with np.load(directory / "spectra.npz") as arrays:
        expected = (3, 180)
        for key in ("float32", "float64", "analytic"):
            if arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid {key} spectra for {name}")
    return record


def benchmark(path):
    report = json.loads(path.read_text())
    result = next(item for item in report["results"] if item["size"] == 512)
    return dict(
        path=str(path),
        dtype=report["configuration"].get("dtype", "float64"),
        cuda_wall_seconds=result["cuda"]["wall_seconds_measured"],
        numpy_wall_seconds=result["numpy"]["wall_seconds_measured"],
        normalized_far_field_relative_l2=result["comparison"]["normalized_far_field_relative_l2"],
    )


def main():
    root = Path("runs/cuda_precision")
    cases = {name: load_case(root, name) for name in CASES}
    benchmarks = {}
    for scene, float64_path, float32_path in (
        (
            "dielectric",
            Path("runs/cuda_foundation/report.json"),
            Path("runs/cuda_foundation_float32/report.json"),
        ),
        (
            "pec_enlarged",
            Path("runs/cuda_pec/report.json"),
            Path("runs/cuda_pec_float32/report.json"),
        ),
    ):
        float64 = benchmark(float64_path)
        float32 = benchmark(float32_path)
        benchmarks[scene] = dict(
            float64=float64,
            float32=float32,
            float32_speedup_vs_float64=float64["cuda_wall_seconds"] / float32["cuda_wall_seconds"],
        )
    all_accuracy_passed = all(record["decision"] == "accepted" for record in cases.values())
    any_throughput_gain = any(
        item["float32_speedup_vs_float64"] > 1 for item in benchmarks.values()
    )
    report = dict(
        schema_version=1,
        accuracy_decision="accepted" if all_accuracy_passed else "rejected",
        production_dtype="float64",
        production_decision=(
            "Float32 passes the declared complex-field, phase, analytic-degradation, and "
            "settling gates, but is slower in both 512x512 throughput probes."
        ),
        any_measured_float32_throughput_gain=any_throughput_gain,
        cases=cases,
        benchmarks=benchmarks,
    )
    atomic_json(root / "report.json", report)
    print(
        f"accuracy={report['accuracy_decision']} production_dtype={report['production_dtype']} "
        f"float32_throughput_gain={any_throughput_gain}"
    )


if __name__ == "__main__":
    main()
