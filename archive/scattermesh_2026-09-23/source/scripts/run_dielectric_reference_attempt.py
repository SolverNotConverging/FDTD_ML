#!/usr/bin/env python3
"""Run and atomically retain one CUDA dielectric reference-escalation attempt."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scattermesh import Circle, Grid, Material, PlaneWave, simulate_cuda
from scattermesh.analytic import cylinder_far_field
from scattermesh.constants import C0
from scattermesh.metrics import compare_far_fields, scattering_loss

FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
SCENES = {
    "eps12_small": dict(
        radius=0.05, center=(0.6387, 0.5669), angle=1.17, epsilon_r=12.0, sigma_e=0.0
    ),
    "eps30_small": dict(
        radius=0.045, center=(0.5431, 0.6537), angle=3.91, epsilon_r=30.0, sigma_e=0.0
    ),
    "eps30_lossy": dict(
        radius=0.045, center=(0.5817, 0.6329), angle=5.20, epsilon_r=30.0, sigma_e=0.2
    ),
}


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }


def fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def cached(record_path, spectra_path, expected_fingerprint):
    if not record_path.exists() or not spectra_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        with np.load(spectra_path) as arrays:
            numerical = arrays["complex_numerical"]
            analytic = arrays["complex_analytic"]
            valid = (
                record.get("fingerprint") == expected_fingerprint
                and numerical.shape == (len(FREQUENCIES), len(ANGLES))
                and analytic.shape == numerical.shape
                and np.iscomplexobj(numerical)
                and np.iscomplexobj(analytic)
                and np.isfinite(numerical).all()
                and np.isfinite(analytic).all()
                and np.array_equal(arrays["frequencies"], FREQUENCIES)
                and np.array_equal(arrays["angles"], ANGLES)
            )
        return record if valid else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--duration-ns", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--pml", type=float, default=0.15)
    parser.add_argument("--larger-contour", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/dielectric_reference_escalation")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cells < 16 or args.duration_ns <= 0 or args.samples < 1 or args.pml <= 0:
        raise ValueError("Positive duration/PML/samples and at least 16 cells per axis required")
    scene = SCENES[args.scene]
    contour = (0.23, 0.97, 0.23, 0.97) if args.larger_contour else (0.3, 0.9, 0.3, 0.9)
    config = dict(
        scene=args.scene,
        **{key: list(value) if key == "center" else value for key, value in scene.items()},
        cells=args.cells,
        duration_ns=args.duration_ns,
        device=args.device,
        samples=args.samples,
        pml=args.pml,
        contour=list(contour),
        frequencies_hz=FREQUENCIES.tolist(),
        observation_angles=ANGLES.tolist(),
        source=dict(frequency_hz=1e9, width_s=1e-9, delay_s=9e-9, origin_m=[0.6, 0.6]),
        source_sha256=source_sha256(),
    )
    attempt_fingerprint = fingerprint(config)
    contour_name = "large" if args.larger_contour else "default"
    attempt_id = (
        f"cells{args.cells}_dur{args.duration_ns:g}ns_s{args.samples}_"
        f"pml{args.pml:g}_{contour_name}_{attempt_fingerprint[:12]}"
    )
    output = args.output_root / args.scene / attempt_id
    output.mkdir(parents=True, exist_ok=True)
    record_path, spectra_path = output / "record.json", output / "spectra.npz"
    record = None if args.force else cached(record_path, spectra_path, attempt_fingerprint)
    if record is not None:
        print(f"{args.scene}: resumed {record['status']} attempt {output}", flush=True)
        return

    axis = np.linspace(0, 1.2, args.cells + 1)
    grid = Grid(axis, axis)
    material = Material(scene["epsilon_r"], scene["sigma_e"])
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=scene["angle"], origin=(0.6, 0.6))
    result = simulate_cuda(
        grid,
        [Circle(scene["center"], scene["radius"], material)],
        source,
        frequencies=FREQUENCIES,
        duration=args.duration_ns * 1e-9,
        pml_thickness=args.pml,
        monitor_bounds=contour,
        samples=args.samples,
        device=args.device,
        dtype="float64",
    )
    numerical = result.monitor.normalized_far_field(ANGLES)
    analytic = np.array(
        [
            cylinder_far_field(
                scene["radius"],
                material,
                frequency,
                ANGLES,
                scene["angle"],
                center=scene["center"],
                incident_origin=source.origin,
            )
            for frequency in FREQUENCIES
        ]
    )
    size = 2 * np.pi * FREQUENCIES.max() / C0 * scene["radius"]
    analytic_order = max(int(np.ceil(size + 4 * size ** (1 / 3) + 8)), 12)
    analytic_high_order = np.array(
        [
            cylinder_far_field(
                scene["radius"],
                material,
                frequency,
                ANGLES,
                scene["angle"],
                center=scene["center"],
                incident_origin=source.origin,
                order=analytic_order + 8,
            )
            for frequency in FREQUENCIES
        ]
    )
    series_error = np.linalg.norm(analytic_high_order - analytic, axis=1) / np.maximum(
        np.linalg.norm(analytic_high_order, axis=1), 1e-30
    )
    numerical_width = 2 * np.pi * abs(numerical) ** 2
    analytic_width = 2 * np.pi * abs(analytic) ** 2
    width_error = np.linalg.norm(numerical_width - analytic_width, axis=1) / np.maximum(
        np.linalg.norm(analytic_width, axis=1), 1e-30
    )
    comparison = compare_far_fields(numerical, analytic)
    loss = scattering_loss(numerical, analytic)
    measured = dict(
        max_complex_relative_l2=max(comparison["complex_relative_l2_by_frequency"]),
        max_phase_rms_degrees=max(value or 0 for value in comparison["phase_weighted_rms_degrees"]),
        tail_peak_over_global_peak=result.diagnostics["tail_peak_over_global_peak"],
        max_analytic_series_relative_l2=float(max(series_error)),
    )
    thresholds = dict(
        max_complex_relative_l2=0.02,
        max_phase_rms_degrees=1.5,
        tail_peak_over_global_peak=1e-5,
        max_analytic_series_relative_l2=1e-10,
    )
    gates = {
        name: dict(measured=value, threshold=thresholds[name], passed=value < thresholds[name])
        for name, value in measured.items()
    }
    finite_fields = all(np.isfinite(field).all() for field in result.fields.values())
    record = dict(
        fingerprint=attempt_fingerprint,
        attempt_id=attempt_id,
        config=config,
        status="pass"
        if finite_fields and all(gate["passed"] for gate in gates.values())
        else "fail",
        finite_fields=finite_fields,
        minimum_internal_cells_per_wavelength=C0
        / (FREQUENCIES.max() * np.sqrt(scene["epsilon_r"]) * np.diff(axis).max()),
        analytic_order=analytic_order,
        analytic_series_relative_l2_by_frequency=series_error.tolist(),
        width_relative_l2_by_frequency=width_error.tolist(),
        gates=gates,
        **comparison,
        **loss,
        **result.diagnostics,
    )
    atomic_npz(
        spectra_path,
        frequencies=FREQUENCIES,
        angles=ANGLES,
        complex_numerical=numerical,
        complex_analytic=analytic,
        incident_spectrum=result.monitor.incident,
        width_numerical=numerical_width,
        width_analytic=analytic_width,
        x=grid.x,
        y=grid.y,
    )
    atomic_json(record_path, record)
    print(
        f"{args.scene}: {record['status']} max_complex={measured['max_complex_relative_l2']:.4g} "
        f"tail={measured['tail_peak_over_global_peak']:.3g} time={record['wall_seconds']:.2f}s "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
