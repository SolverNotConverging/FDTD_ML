"""Compare outgoing wave packets against a larger domain before its echoes return."""

import argparse
import json
from pathlib import Path

import numpy as np

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import C0, EPS0, MU0


def initial_packet(mesh, dt, center, angle):
    theta = np.deg2rad(angle)
    nx, ny = np.cos(theta), np.sin(theta)
    k = 2 * np.pi * 60e9 / C0
    width = 0.0015

    def packet(x, y, time):
        X, Y = np.meshgrid(x - center[0], y - center[1], indexing="ij")
        longitudinal = nx * X + ny * Y - C0 * time
        transverse = -ny * X + nx * Y
        return np.exp(-(longitudinal**2 + transverse**2) / width**2) * np.cos(k * longitudinal)

    x, y = mesh.x, mesh.y
    impedance = np.sqrt(MU0 / EPS0)
    return {
        "Ez": packet(x, y, 0),
        "Hx": ny / impedance * packet(x, (y[:-1] + y[1:]) / 2, -dt / 2),
        "Hy": -nx / impedance * packet((x[:-1] + x[1:]) / 2, y, -dt / 2),
    }


def extend_axis(lines, pad, spacing):
    offset = pad * spacing
    # The original mesh is retained exactly up to the coordinate translation.
    return np.r_[
        np.arange(pad) * spacing,
        lines + offset,
        lines[-1] + offset + np.arange(1, pad + 1) * spacing,
    ]


def wavepacket_case(angle=0, nonuniform=False, *, long_run=False, pec_control=False):
    n, length, pml_cells = 96, 0.024, 12
    h = length / n
    s = FDTD_2D_Ez(length, length, n, n, 100e9, t_end=1.6e-10, dtype="float64")
    s.add_PML(pml_cells, thickness=pml_cells * h, kappa_max=3, alpha_max=0.05)
    probes = [(length / 2, length / 2)] + [
        (length / 2 + 0.004 * np.cos(t), length / 2 + 0.004 * np.sin(t))
        for t in np.arange(8) * np.pi / 4
    ]
    for x, y in probes:
        s.add_receiver("point", x=x, y=y)
    density = np.array([1, 1, 1.2, 1.8, 1.8, 1.2, 1, 1]) if nonuniform else [1]
    s.mesh_from_density(density, density)
    c, *_ = s._prepare()
    fields = initial_packet(s.mesh, c.dt, (length / 2, length / 2), angle)
    small = s.run(initial_fields=fields)
    pad = 100
    xr, yr = extend_axis(s.mesh.x, pad, h), extend_axis(s.mesh.y, pad, h)
    reference = FDTD_2D_Ez(
        xr[-1],
        yr[-1],
        len(xr) - 1,
        len(yr) - 1,
        100e9,
        Nt=small.diagnostics["Nt"],
        dt=c.dt,
        dtype="float64",
    )
    reference.set_mesh(xr, yr)
    for x, y in probes:
        reference.add_receiver("point", x=x + pad * h, y=y + pad * h)
    reference_fields = initial_packet(
        reference.mesh, c.dt, (length / 2 + pad * h, length / 2 + pad * h), angle
    )
    large = reference.run(initial_fields=reference_fields)
    a = pml_cells
    interior = (slice(a, n - a + 1), slice(a, n - a + 1))
    reference_crop = large.fields["Ez"][pad : pad + n + 1, pad : pad + n + 1]
    norm = np.linalg.norm(fields["Ez"][interior])
    difference = small.fields["Ez"] - reference_crop
    error = np.linalg.norm(difference[interior]) / norm
    history = np.column_stack([small.receivers[i][:, 0] for i in range(len(probes))])
    ref_history = np.column_stack([large.receivers[i][:, 0] for i in range(len(probes))])
    delta = history - ref_history
    echo_peak = np.max(abs(delta)) / np.max(abs(ref_history))
    echo_l2 = np.linalg.norm(delta) / np.linalg.norm(ref_history)
    report = {
        "angle_degrees": angle,
        "nonuniform": nonuniform,
        "relative_interior_echo": float(error),
        "peak_waveform_error": float(echo_peak),
        "relative_waveform_l2": float(echo_l2),
        "peak_echo_db": float(20 * np.log10(max(echo_peak, 1e-30))),
        "Nt": small.diagnostics["Nt"],
        "dt": small.dt,
        "gpu_ms": small.diagnostics["gpu_ms"],
        "stepping_transfers": small.diagnostics["stepping_transfers"],
    }
    if pec_control:
        control = FDTD_2D_Ez(
            length, length, n, n, 100e9, Nt=small.diagnostics["Nt"], dt=c.dt, dtype="float64"
        )
        control.set_mesh(s.mesh.x, s.mesh.y)
        for x, y in probes:
            control.add_receiver("point", x=x, y=y)
        reflected = control.run(initial_fields=fields)
        control_history = np.column_stack(
            [reflected.receivers[i][:, 0] for i in range(len(probes))]
        )
        report["pec_control_peak_error"] = float(
            np.max(abs(control_history - ref_history)) / np.max(abs(ref_history))
        )
    if long_run:
        s.t_end = 1.5e-9
        late = s.run(initial_fields=fields)
        report["late_relative_field"] = float(np.linalg.norm(late.fields["Ez"][interior]) / norm)
        report["late_finite"] = bool(all(np.isfinite(f).all() for f in late.fields.values()))
    traces = {"times": small.times, "small": history, "reference": ref_history, "difference": delta}
    return report, s.mesh, fields["Ez"], small.fields["Ez"], difference, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for nonuniform in (False, True):
        for angle in (0, 30, 45, 90):
            report, *_ = wavepacket_case(
                angle, nonuniform, long_run=angle == 45, pec_control=angle == 45
            )
            print(json.dumps(report), flush=True)
            records.append(report)
    (args.output_dir / "cpml_validation.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
