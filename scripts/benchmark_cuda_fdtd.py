#!/usr/bin/env python3
"""Paired same-GPU benchmark of the legacy Torch and compiled CUDA FDTD loops."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from scattermesh import PEC, Circle, Grid, Material, PlaneWave, Rectangle, simulate_cuda
from scattermesh.candidates_v2 import candidate_axes
from scattermesh.curriculum_v2 import generate_development_scenes, objects_from_scene
from scattermesh.monitor_v2 import widest_non_pml_monitor_bounds

FREQUENCIES = (0.8e9, 1e9, 1.2e9)
ANGLES = np.linspace(0, 2 * np.pi, 72, endpoint=False)


def cases():
    scene = next(row for row in generate_development_scenes() if row["stage"] == "C5")
    x, y, _ = candidate_axes(scene, 64, "uniform")
    return (
        (
            "dielectric_circle_uniform_96",
            Grid(np.linspace(0, 1.2, 97), np.linspace(0, 1.2, 97)),
            [Circle((0.6, 0.6), 0.06, Material(4, 0.01))],
            70e-9,
        ),
        (
            "pec_rectangle_conformal_32",
            Grid(np.linspace(0, 1.2, 33), np.linspace(0, 1.2, 33)),
            [Rectangle((0.388, 0.812, 0.401, 0.799), PEC())],
            12e-9,
        ),
        ("distributed_dielectric_64", Grid(x, y, max_ratio=3), objects_from_scene(scene), 70e-9),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("runs_v2/cuda_fdtd_benchmark_v1/report.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not visible to this process")
    device = torch.device(args.device)
    source = PlaneWave(1e9, 1e-9, 9e-9, origin=(0.6, 0.6))
    report = {
        "schema_version": 1,
        "protocol": "three paired repeats per case, alternating backend order; wall time includes setup and output copies",
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "repeats": args.repeats,
        "kernel_build": json.loads(
            (
                Path(__file__).resolve().parents[1] / "src/scattermesh/_cuda_fdtd_build.json"
            ).read_text()
        ),
        "cases": [],
    }
    for name, grid, objects, duration in cases():
        monitor = widest_non_pml_monitor_bounds(grid, [obj.bounds for obj in objects], 0.12)
        runs = {"torch": [], "compiled": []}
        fields = {}
        for repeat in range(args.repeats):
            order = ("torch", "compiled") if repeat % 2 == 0 else ("compiled", "torch")
            for backend in order:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                result = simulate_cuda(
                    grid,
                    objects,
                    source,
                    frequencies=FREQUENCIES,
                    duration=duration,
                    pml_thickness=0.12,
                    monitor_bounds=monitor,
                    samples=8,
                    device=args.device,
                    dtype="float64",
                    pec_mode="conformal",
                    backend=backend,
                )
                torch.cuda.synchronize(device)
                fields[(repeat, backend)] = result.monitor.normalized_far_field(ANGLES)
                runs[backend].append(
                    {
                        "wall_seconds": result.diagnostics["wall_seconds"],
                        "stepping_seconds": result.diagnostics["stepping_seconds"],
                        "steps": result.diagnostics["Nt"],
                        "cuda_memory_bytes": result.diagnostics.get(
                            "peak_cuda_memory_bytes", torch.cuda.max_memory_allocated(device)
                        ),
                        "host_transfers_during_steps": result.diagnostics.get(
                            "host_transfers_during_steps"
                        ),
                    }
                )
        torch_wall = statistics.median(row["wall_seconds"] for row in runs["torch"])
        compiled_wall = statistics.median(row["wall_seconds"] for row in runs["compiled"])
        torch_step = statistics.median(row["stepping_seconds"] for row in runs["torch"])
        compiled_step = statistics.median(row["stepping_seconds"] for row in runs["compiled"])
        differences = [
            float(
                np.linalg.norm(fields[(r, "compiled")] - fields[(r, "torch")])
                / max(np.linalg.norm(fields[(r, "torch")]), 1e-30)
            )
            for r in range(args.repeats)
        ]
        row = {
            "name": name,
            "cells": [len(grid.x) - 1, len(grid.y) - 1],
            "duration_s": duration,
            "steps": runs["torch"][0]["steps"],
            "median_torch_wall_seconds": torch_wall,
            "median_compiled_wall_seconds": compiled_wall,
            "wall_speedup": torch_wall / compiled_wall,
            "median_torch_step_seconds": torch_step,
            "median_compiled_step_seconds": compiled_step,
            "step_speedup": torch_step / compiled_step,
            "maximum_complex_far_field_relative_l2": max(differences),
            "runs": runs,
        }
        report["cases"].append(row)
        print(name, "wall speedup", round(row["wall_speedup"], 1), "x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
