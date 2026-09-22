"""Diagnostic only: exact slab-interface averaging vs current Yee point sampling.

A PEC rectangular cavity has epsilon=1 to x=a and epsilon=contrast to x=Lx.
Ez is tangential to the dielectric interface; its dual-cell epsilon is the
arithmetic area mean. Separation in y reduces the unchanged Yee curl operator
to a symmetric generalized tridiagonal eigenproblem. Leapfrog dispersion is
included. The continuum reference is an independent transfer-matrix root.
No production solver behavior is changed.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import C0, EPS0, MU0
from fdtdmesh.solver.coefficients import build_coefficients
from fdtdmesh.solver.reference_tmz import run_reference

LX, LY = 0.02, 0.015


def analytic_frequency(fraction, contrast):
    a, b = fraction * LX, (1 - fraction) * LX

    def residual(q):
        k1 = np.sqrt(complex(q * q - (np.pi / LY) ** 2))
        k2 = np.sqrt(complex(contrast * q * q - (np.pi / LY) ** 2))
        return float(
            (
                np.cos(k2 * b) * a * np.sinc(k1 * a / np.pi)
                + b * np.sinc(k2 * b / np.pi) * np.cos(k1 * a)
            ).real
        )

    # Fundamental eigenvalue lies between homogeneous contrast and vacuum bounds.
    base = np.pi * np.sqrt(LX**-2 + LY**-2)
    q = np.linspace(base / np.sqrt(contrast) * 0.999, base * 1.001, 10001)
    values = np.array([residual(v) for v in q])
    i = np.flatnonzero(values[:-1] * values[1:] <= 0)[0]
    return C0 * brentq(residual, q[i], q[i + 1], xtol=1e-12) / (2 * np.pi)


def case(n, fraction, contrast, verify=False):
    s = FDTD_2D_Ez(LX, LY, n, n, 20e9, Nt=1, dtype="float64")
    s.add_material("dielectric", epsilon_r=contrast)
    s.add_rectangle("dielectric", (fraction * LX, LX), (0, LY))
    s.mesh_uniform()
    c = build_coefficients(s, s.mesh, dtype="float64")
    x, y = s.mesh.x, s.mesh.y
    dx, dy = LX / n, LY / n
    original = c.dt / (EPS0 * c.current_scale[:, n // 2])
    # Exact rectangular dual-cell fractions, including clipped domain edges.
    left, right = np.maximum(0, x - dx / 2), np.minimum(LX, x + dx / 2)
    fill = np.clip((right - np.maximum(left, fraction * LX)) / (right - left), 0, 1)
    averaged = 1 + (contrast - 1) * fill
    truth = analytic_frequency(fraction, contrast)
    rows = []
    for name, eps in [("point", original), ("averaged", averaged)]:
        e = eps[1:-1]
        ky2 = (2 * np.sin(np.pi / (2 * n)) / dy) ** 2
        diagonal = (2 / dx**2 + ky2) / e
        off = -1 / dx**2 / np.sqrt(e[:-1] * e[1:])
        eigen, v = eigh_tridiagonal(diagonal, off, select="i", select_range=(0, 0))
        omega_discrete = 2 / c.dt * np.arcsin(c.dt * C0 * np.sqrt(eigen[0]) / 2)
        frequency = omega_discrete / (2 * np.pi)
        row = dict(
            n=n,
            mapping=name,
            interface_fraction=fraction,
            contrast=contrast,
            frequency_hz=frequency,
            analytic_hz=truth,
            relative_error=abs(frequency / truth - 1),
        )
        if verify:
            shape = np.r_[0, v[:, 0] / np.sqrt(e), 0]
            ez = shape[:, None] * np.sin(np.pi * y[None, :] / LY)
            ez /= np.max(abs(ez))
            ez[[0, -1], :] = 0
            ez[:, [0, -1]] = 0
            hx = c.dt / (2 * MU0) * np.diff(ez, axis=1) / dy
            hy = -c.dt / (2 * MU0) * np.diff(ez, axis=0) / dx
            ratio = original / eps
            updated = replace(
                c,
                cbx=np.ascontiguousarray(c.cbx * ratio[:, None]),
                cby=np.ascontiguousarray(c.cby * ratio[:, None]),
                current_scale=np.ascontiguousarray(c.current_scale * ratio[:, None]),
            )
            nt = int(np.ceil(2 / frequency / c.dt))
            fields = run_reference(
                updated,
                (ez, hx, hy),
                np.array([], dtype=np.int64),
                np.zeros((nt, 0)),
                np.array([], dtype=np.int64),
            )
            error = np.max(abs(fields[0] - ez * np.cos(omega_discrete * nt * c.dt)))
            row["numpy_time_evolution_max_error"] = float(error)
            assert error < 1e-8, error
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for contrast, fraction in [(1, 0.371), (12, 0.5), (12, 0.371), (30, 0.371)]:
        for n in [32, 64, 128, 256, 512]:
            rows.extend(case(n, fraction, contrast, verify=(n == 64)))
        print("Completed", contrast, fraction, flush=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (contrast, fraction) in zip(
        axes.flat, [(1, 0.371), (12, 0.5), (12, 0.371), (30, 0.371)]
    ):
        for mapping, label in [
            ("point", "Current point sampling"),
            ("averaged", "Exact dual-cell averaging"),
        ]:
            selected = [
                r
                for r in rows
                if r["contrast"] == contrast
                and r["interface_fraction"] == fraction
                and r["mapping"] == mapping
            ]
            ax.loglog(
                [r["n"] for r in selected],
                [r["relative_error"] * 100 for r in selected],
                "o-",
                label=label,
            )
        ax.set(
            title=f"εr = {contrast}, interface x/Lx = {fraction}",
            xlabel="Cells per axis",
            ylabel="Frequency error vs analytical [%]",
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Controlled Yee-interface test: same geometry, timestep and curl stencil\nμr = 1, σe = σh = 0; analytical slab-cavity reference"
    )
    fig.savefig(args.output / "convergence.png", dpi=170)
    plt.close(fig)
    print(json.dumps([r for r in rows if r["n"] == 512], indent=2))


if __name__ == "__main__":
    main()
