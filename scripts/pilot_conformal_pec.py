#!/usr/bin/env python3
"""Off-grid PEC pilot: analytical cylinder accuracy and small-cut stability cost."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch

from scattermesh import PEC, Circle, Grid, PlaneWave, Rectangle, focused_axis, simulate
from scattermesh.analytic import cylinder_far_field, cylinder_width
from scattermesh.conformal import CutCellPEC
from scattermesh.metrics import compare_far_fields, scattering_loss


def plot_complex_spectra(output):
    """Plot cached complex spectra without rerunning a simulation."""
    with np.load(Path(output) / "spectra.npz") as data:
        degrees = np.rad2deg(data["angles"])
        reference = data["circle_n64_enlarged_analytic_complex_far_field"][1]
        predictions = [
            ("Uniform 64²", data["circle_n64_enlarged_complex_far_field"][1]),
            ("Nonuniform 64²", data["circle_graded_n64_enlarged_complex_far_field"][1]),
        ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, part, label in zip(axes[:2], [np.real, np.imag], ["Re(F)", "Im(F)"]):
        ax.plot(degrees, part(reference), "k--", label="Analytic")
        for name, prediction in predictions:
            ax.plot(degrees, part(prediction), label=name)
        ax.set(xlabel="Observation angle (degrees)", ylabel=label + " (sqrt(m))")
        ax.grid(alpha=0.2)
        ax.legend()
    valid = abs(reference) >= 0.05 * abs(reference).max()
    for name, prediction in predictions:
        error = np.rad2deg(np.angle(prediction * np.conj(reference)))
        axes[2].plot(degrees, np.where(valid, error, np.nan), label=name)
    axes[2].set(
        xlabel="Observation angle (degrees)",
        ylabel="Phase error (degrees)",
        title="No fitted phase rotation\nReference nulls excluded",
    )
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.suptitle("Complex far field: enlarged PEC, 1 GHz, incidence 40.1°")
    fig.savefig(Path(output) / "complex_far_field.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/conformal_pec_pilot"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    frequencies = [0.8e9, 1e9, 1.2e9]
    records, spectra = [], {}

    def run(name, grid, obj, angle, mode="conformal", duration=35e-9):
        source = PlaneWave(1e9, 1e-9, 9e-9, angle=angle, origin=(0.6, 0.6))
        result = simulate(
            grid,
            [obj],
            source,
            frequencies=frequencies,
            duration=duration,
            pml_thickness=0.15,
            pec_mode=mode,
        )
        width = result.monitor.scattering_width(angles)
        field = result.monitor.normalized_far_field(angles)
        record = dict(
            name=name,
            **result.diagnostics,
            finite_fields=all(np.isfinite(f).all() for f in result.fields.values()),
        )
        if isinstance(obj, Circle):
            reference = np.array(
                [cylinder_width(obj.radius, PEC(), f, angles, angle) for f in frequencies]
            )
            error = np.linalg.norm(width - reference, axis=1) / np.linalg.norm(reference, axis=1)
            record["width_relative_l2_by_frequency"] = error.tolist()
            reference_field = np.array(
                [
                    cylinder_far_field(
                        obj.radius,
                        PEC(),
                        f,
                        angles,
                        angle,
                        center=obj.center,
                        incident_origin=source.origin,
                    )
                    for f in frequencies
                ]
            )
            record.update(compare_far_fields(field, reference_field))
            record.update(scattering_loss(field, reference_field))
            spectra[name + "_analytic_complex_far_field"] = reference_field
            record["geometry"] = dict(kind="circle", center=list(obj.center), radius=obj.radius)
        else:
            record["geometry"] = dict(kind="rectangle", bounds=list(obj.bounds))
        spectra[name] = width
        spectra[name + "_complex_far_field"] = field
        spectra[name + "_incident_spectrum"] = result.monitor.incident.copy()
        records.append(record)
        (args.output / "partial_records.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            f"{name}: Nt={record['Nt']}, dt/grid={record['dt_fraction_of_grid_cfl']:.4f}, "
            f"min_cut={record['pec_minimum_open_fraction']:.5f}, "
            f"tail={record['tail_peak_over_global_peak']:.3g}, "
            f"seconds={record['wall_seconds']:.2f}",
            flush=True,
        )

    circle = Circle((0.613, 0.591), 0.08, PEC())
    for cells in [48, 64, 96, 128]:
        x = focused_axis(1.2, cells)
        for mode in ["staircase", "conformal", "enlarged"]:
            run(f"circle_n{cells}_{mode}", Grid(x, x), circle, 0.7, mode)
    for cells in [48, 64, 96]:
        x = focused_axis(1.2, cells, width=0.12, strength=2)
        for mode in ["staircase", "conformal", "enlarged"]:
            run(f"circle_graded_n{cells}_{mode}", Grid(x, x), circle, 0.7, mode)
    x = focused_axis(1.2, 64)
    for degrees in [0, 113, 227]:
        run(f"circle_angle{degrees}", Grid(x, x), circle, np.deg2rad(degrees), "enlarged")
    # Same circle moved within a single cell: physical width is translation invariant.
    for offset in [(0.0021, -0.0013), (-0.0042, 0.0037)]:
        shifted = Circle((circle.center[0] + offset[0], circle.center[1] + offset[1]), 0.08, PEC())
        run(f"circle_shift{offset[0]}", Grid(x, x), shifted, 0.7, "enlarged")
    x = focused_axis(1.2, 48)
    for fraction in [0.2, 0.02, 0.002]:
        # The left PEC edge is fraction*dx from a vacuum node; never snapped.
        box = Rectangle((0.475 + fraction * 0.025, 0.71025, 0.4873, 0.6891), PEC())
        for mode in ["conformal", "enlarged"]:
            run(f"box_fraction{fraction}_{mode}", Grid(x, x), box, 0.7, mode, duration=60e-9)
    # Stress a cut only one millionth of a cell from a node. The plain scheme's
    # Nt would exceed the pilot limit; enlargement is explicitly tested here.
    box = Rectangle((0.475 + 1e-6 * 0.025, 0.71025, 0.4873, 0.6891), PEC())
    run("box_fraction1e-6_enlarged", Grid(x, x), box, 0.7, "enlarged", duration=60e-9)
    x = focused_axis(1.2, 64, width=0.12, strength=2)
    for mode in ["conformal", "enlarged"]:
        run(
            "box_graded_" + mode,
            Grid(x, x),
            Rectangle((0.4781, 0.7113, 0.4917, 0.6851), PEC()),
            0.7,
            mode,
            duration=60e-9,
        )

    report = dict(
        schema=2,
        primary_target="source-normalized complex far field A/Einc, sqrt(m)",
        complex_storage="native complex128 in spectra.npz; real/imaginary parts must be retained",
        phase_convention="exp(-i omega t), outgoing exp(+i k r), global far-field origin (0,0)",
        created_utc=datetime.now(timezone.utc).isoformat(),
        formulation="TMz cut-edge PEC with unchanged Yee electric dual metrics",
        time_step="min(background CFL, conservative cut-edge Gershgorin bound), safety=0.9",
        frequencies_hz=frequencies,
        domain_m=[1.2, 1.2],
        records=records,
        source_sha256={
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted((Path(__file__).resolve().parents[1] / "src/scattermesh").glob("*.py"))
        },
        limitations=[
            "CPU TMz only, no general 3D cut faces",
            "Mixed PEC/dielectric and split-edge thin screens not implemented",
            "Rectangle runs check stability/settling, not analytic accuracy",
            "Finite tests do not prove all offsets/grading stable with CPML",
            "Enlargement is a local TMz Galerkin aggregation, not a general 3D algorithm",
            "An independent CFL bound is still enforced after enlargement",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "spectra.npz", angles=angles, frequencies=frequencies, **spectra
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, strength, title in zip(axes[:2], [0, 2], ["Uniform 64 × 64", "Nonuniform 64 × 64"]):
        x = focused_axis(1.2, 64, width=0.12, strength=strength)
        cut = CutCellPEC(Grid(x, x), [circle])
        for coordinate in x:
            ax.axvline(coordinate, color=".65", lw=0.45)
            ax.axhline(coordinate, color=".65", lw=0.45)
        ax.add_patch(CirclePatch(circle.center, circle.radius, fc=".25", ec="k", alpha=0.7))
        for boundary in cut.boundaries:
            if boundary is not None:
                ax.scatter(boundary[4], boundary[5], s=9, color="tab:red", zorder=3)
        ax.set(
            xlim=(0.46, 0.76),
            ylim=(0.44, 0.74),
            xlabel="x (m)",
            ylabel="y (m)",
            title=title + "\nPEC boundary crosses unchanged mesh",
        )
        ax.set_aspect("equal")
    ax = axes[2]
    for mode, label in [
        ("staircase", "Staircase"),
        ("conformal", "Conformal, reduced dt"),
        ("enlarged", "Enlarged conformal"),
    ]:
        subset = [
            next(r for r in records if r["name"] == f"circle_n{n}_{mode}")
            for n in [48, 64, 96, 128]
        ]
        ax.plot(
            [r["Nx"] for r in subset],
            [100 * r["width_relative_l2_by_frequency"][1] for r in subset],
            "o-",
            label=label,
        )
    ax.set(
        xlabel="Cells per axis",
        ylabel="Angular scattering-width L2 error (%)",
        title="PEC cylinder vs analytic solution\n1 GHz, incidence 40.1°",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(args.output / "conformal_pec.png", dpi=160)
    plt.close(fig)
    plot_complex_spectra(args.output)
    print("Report:", args.output / "report.json")


if __name__ == "__main__":
    main()
