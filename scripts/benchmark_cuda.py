#!/usr/bin/env python3
"""CUDA throughput benchmark (active source at 10 ns; not an accuracy qualification)."""

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scattermesh import PEC, Circle, Grid, Material, PlaneWave, simulate, simulate_cuda

ANGLES = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])


def _relative_l2(a, b):
    return float(
        np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(a)), 1e-30)
    )


def _environment(device):
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        result.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda})
        if torch.cuda.is_available():
            index = torch.device(device).index or 0
            props = torch.cuda.get_device_properties(index)
            result["nvidia_device"] = {
                "name": props.name,
                "index": index,
                "capability": [props.major, props.minor],
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
    except ImportError:
        result["torch"] = None
    try:
        result["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git"] = None
    return result


def _case(size, duration, device, scene):
    grid = Grid(np.linspace(0.0, 1.2, size + 1), np.linspace(0.0, 1.2, size + 1))
    material = Material(12.0) if scene == "dielectric" else PEC()
    objects = [Circle((0.6387, 0.5669), 0.05, material)]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=1.17, origin=(0.6, 0.6))
    settings = dict(frequencies=FREQUENCIES, duration=duration, pml_thickness=0.15, samples=24)
    if scene == "pec_enlarged":
        settings["pec_mode"] = "enlarged"
    started = time.perf_counter()
    reference = simulate(grid, objects, source, **settings)
    numpy_wall = time.perf_counter() - started
    started = time.perf_counter()
    candidate = simulate_cuda(grid, objects, source, device=device, dtype="float64", **settings)
    cuda_wall = time.perf_counter() - started
    field_errors = {
        name: _relative_l2(reference.fields[name], candidate.fields[name])
        for name in ("Ez", "Hx", "Hy")
    }
    monitor_errors = {
        name: _relative_l2(getattr(reference.monitor, name), getattr(candidate.monitor, name))
        for name in ("electric", "tangential_h", "incident")
    }
    far_reference = reference.monitor.normalized_far_field(ANGLES)
    far_candidate = candidate.monitor.normalized_far_field(ANGLES)
    result = {
        "size": size,
        "numpy": {**reference.diagnostics, "wall_seconds_measured": numpy_wall, "dtype": "float64"},
        "cuda": {**candidate.diagnostics, "wall_seconds_measured": cuda_wall},
        "comparison": {
            "final_fields_relative_l2": field_errors,
            "monitor_relative_l2": monitor_errors,
            "normalized_far_field_relative_l2": _relative_l2(far_reference, far_candidate),
        },
    }
    return result, numpy_wall, cuda_wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/cuda_foundation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--duration", type=float, default=10e-9)
    parser.add_argument("--scene", choices=["dielectric", "pec_enlarged"], default="dielectric")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    for size in args.sizes:
        row, numpy_wall, cuda_wall = _case(size, args.duration, args.device, args.scene)
        row["cuda"]["speedup_vs_numpy_wall"] = numpy_wall / max(cuda_wall, 1e-30)
        row["cuda"]["cell_updates_per_second"] = row["cuda"]["cell_updates"] / max(cuda_wall, 1e-30)
        rows.append(row)
        times.append((size, numpy_wall, cuda_wall))
    root = Path(__file__).resolve().parents[1]
    source_hash = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), *sorted((root / "src/scattermesh").glob("*.py"))]
    }
    report = {
        "benchmark": "CUDA throughput; active source at 10 ns; not an accuracy qualification",
        "configuration": {
            "device": args.device,
            "sizes": args.sizes,
            "duration": args.duration,
            "frequencies_hz": FREQUENCIES.tolist(),
            "domain": [1.2, 1.2],
            "pml": 0.15,
            "samples": 24,
            "scene": args.scene,
        },
        "source_sha256": source_hash,
        "environment": _environment(args.device),
        "results": rows,
    }
    report_path = args.output / "report.json"
    with tempfile.NamedTemporaryFile(
        "w", dir=args.output, prefix="report.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(report_path)
    sizes, numpy_times, cuda_times = zip(*times)
    fig, axis = plt.subplots()
    axis.plot(sizes, numpy_times, "o-", label="NumPy float64")
    axis.plot(sizes, cuda_times, "o-", label=f"CUDA float64 ({args.device})")
    axis.set(
        xlabel="Grid cells per axis", ylabel="Wall time (s)", title="CUDA throughput benchmark"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "size_vs_time_speedup.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
