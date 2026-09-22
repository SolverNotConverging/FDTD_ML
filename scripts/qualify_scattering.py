#!/usr/bin/env python3
"""Reproducible, small CPU qualification; does not launch a reference campaign."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scattermesh import Circle, Grid, Material, PlaneWave, focused_axis, simulate
from scattermesh.analytic import cylinder_width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/scattering_bootstrap"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frequencies = np.array([0.8e9, 1e9, 1.2e9])
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    material = Material(4)
    records, arrays = [], {}
    curves = {}

    def run(
        name,
        cells,
        strength,
        angle,
        *,
        duration=30e-9,
        pml=0.15,
        bounds=(0.3, 0.9, 0.3, 0.9),
        samples=12,
    ):
        axis = focused_axis(1.2, cells, width=0.12, strength=strength)
        grid = Grid(axis, axis)
        source = PlaneWave(1e9, 1e-9, 9e-9, angle=np.deg2rad(angle), origin=(0.6, 0.6))
        result = simulate(
            grid,
            [Circle((0.6, 0.6), 0.06, material)],
            source,
            frequencies=frequencies,
            duration=duration,
            pml_thickness=pml,
            monitor_bounds=bounds,
            samples=samples,
        )
        width = result.monitor.scattering_width(angles)
        reference = np.array(
            [cylinder_width(0.06, material, f, angles, source.angle) for f in frequencies]
        )
        error = np.linalg.norm(width - reference, axis=1) / np.linalg.norm(reference, axis=1)
        records.append(
            dict(
                name=name,
                strength=strength,
                angle_degrees=angle,
                width_relative_l2_by_frequency=error.tolist(),
                **result.diagnostics,
            )
        )
        curves[name] = width
        arrays[name + "_width"] = width
        arrays[name + "_analytic"] = reference
        arrays[name + "_x"] = axis
        print(
            f"{name}: error={100 * error}% Nt={result.diagnostics['Nt']} "
            f"time={result.diagnostics['wall_seconds']:.2f}s",
            flush=True,
        )
        return width

    for cells in [64, 128]:
        for strength in [0, 2]:
            for angle in [0, 30]:
                run(f"n{cells}_focus{strength}_angle{angle}", cells, strength, angle)
    baseline = curves["n128_focus2_angle30"]
    variants = {
        "longer_duration": dict(duration=40e-9),
        "thicker_pml": dict(pml=0.2),
        "larger_contour": dict(bounds=(0.25, 0.95, 0.25, 0.95)),
        "finer_quadrature": dict(samples=24),
    }
    sensitivities = {}
    for name, settings in variants.items():
        width = run(name, 128, 2, 30, **settings)
        sensitivities[name] = (
            np.linalg.norm(width - baseline, axis=1) / np.linalg.norm(baseline, axis=1)
        ).tolist()
    report = dict(
        schema=1,
        purpose="initial CPU dielectric-cylinder qualification, not campaign acceptance",
        created_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        source_sha256={
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted((Path(__file__).resolve().parents[1] / "src/scattermesh").glob("*.py"))
        },
        geometry=dict(
            kind="dielectric_circle",
            center=[0.6, 0.6],
            radius=0.06,
            epsilon_r=4,
            sigma_e=0,
            mu_r=1,
            sigma_h=0,
        ),
        domain_m=[1.2, 1.2],
        frequencies_hz=frequencies.tolist(),
        width_unit="metres (2D scattering width)",
        metric="angular L2(width - analytic) / L2(analytic) at each frequency",
        records=records,
        relative_width_sensitivity=sensitivities,
        limitations=[
            "TMz CPU prototype",
            "This report does not exercise PEC; no CUDA backend yet",
            "One cylinder geometry",
            "This report uses smooth grading; separate tests cover one abrupt ratio 2.8 case",
            "No openEMS numerical cross-run yet",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "spectra.npz", angles=angles, frequencies=frequencies, **arrays
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, cells in zip(axes[:2], [64, 128]):
        key = f"n{cells}_focus2_angle30"
        ax.plot(np.rad2deg(angles), arrays[key + "_analytic"][1], "k--", label="Analytic cylinder")
        for strength, label in [(0, "Uniform"), (2, "Nonuniform")]:
            ax.plot(np.rad2deg(angles), curves[f"n{cells}_focus{strength}_angle30"][1], label=label)
        ax.set(
            xlabel="Observation angle (degrees)",
            ylabel="2D scattering width (m)",
            title=f"{cells} × {cells}, 1 GHz, incidence 30°",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    ax = axes[2]
    for strength, label in [(0, "Uniform"), (2, "Nonuniform")]:
        for cells, style in [(64, "--"), (128, "-")]:
            rec = next(r for r in records if r["name"] == f"n{cells}_focus{strength}_angle30")
            ax.plot(
                frequencies / 1e9,
                100 * np.array(rec["width_relative_l2_by_frequency"]),
                style,
                marker="o",
                label=f"{label}, {cells}²",
            )
    ax.set(
        xlabel="Frequency (GHz)",
        ylabel="Relative angular L2 error (%)",
        title="Refinement against analytic solution",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.output / "scattering_validation.png", dpi=160)
    plt.close(fig)
    print("Report:", args.output / "report.json")


if __name__ == "__main__":
    main()
