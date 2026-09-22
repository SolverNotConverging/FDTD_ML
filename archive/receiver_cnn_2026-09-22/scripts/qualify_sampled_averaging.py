"""Validate production sampled coefficients against analytic lossless/lossy slab modes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from check_interface_averaging import LX, LY, analytic_frequency
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import eigs

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.constants import C0, EPS0
from fdtdmesh.solver.coefficients import build_coefficients


def analytic_mode(fraction, contrast, sigma):
    omega0 = 2 * np.pi * analytic_frequency(fraction, contrast)
    if sigma == 0:
        return complex(omega0)
    a, b = fraction * LX, (1 - fraction) * LX

    def residual(z):
        omega = omega0 * complex(*z)
        k1 = np.sqrt(complex((omega / C0) ** 2 - (np.pi / LY) ** 2))
        k2 = np.sqrt(
            complex(
                (omega / C0) ** 2 * (contrast + 1j * sigma / (EPS0 * omega)) - (np.pi / LY) ** 2
            )
        )
        v = (
            np.cos(k2 * b) * a * np.sinc(k1 * a / np.pi)
            + b * np.sinc(k2 * b / np.pi) * np.cos(k1 * a)
        ) / LX
        return [v.real, v.imag]

    result = root(residual, [1.0, -sigma / (2 * EPS0 * contrast * omega0)], tol=1e-11)
    assert result.success and np.linalg.norm(residual(result.x)) < 1e-9
    return omega0 * complex(*result.x)


def measure(n, contrast, sigma, mode, samples):
    fraction = 0.371
    s = FDTD_2D_Ez(
        LX,
        LY,
        n,
        n,
        20e9,
        Nt=1,
        dtype="float64",
        material_averaging=mode,
        averaging_samples=samples,
        averaging_max_samples=samples,
    )
    s.add_material("a", epsilon_r=contrast, sigma_e=sigma)
    s.add_rectangle("a", (fraction * LX, LX), (0, LY))
    s.mesh_uniform()
    c = build_coefficients(s, s.mesh, dtype="float64")
    loss = (1 - c.ca[:, n // 2]) / (1 + c.ca[:, n // 2])
    eps = c.dt / (EPS0 * c.current_scale[:, n // 2] * (1 + loss))
    sig = 2 * EPS0 * eps * loss / c.dt
    e = eps[1:-1]
    sig = sig[1:-1]
    dx, dy = LX / n, LY / n
    ky2 = (2 * np.sin(np.pi / (2 * n)) / dy) ** 2
    truth = analytic_mode(fraction, contrast, sigma)
    if sigma == 0:
        val = eigh_tridiagonal(
            (2 / dx**2 + ky2) / e,
            -1 / dx**2 / np.sqrt(e[:-1] * e[1:]),
            select="i",
            select_range=(0, 0),
            eigvals_only=True,
        )[0]
        omega = complex(2 / c.dt * np.arcsin(c.dt * C0 * np.sqrt(val) / 2))
    else:
        # E[n+1] = A E[n] + B E[n-1], from leapfrog H + CN electric loss.
        d = e / c.dt**2
        damping = sig / (2 * EPS0 * c.dt)
        laplace = diags(
            [-np.ones(n - 2) / dx**2, np.full(n - 1, 2 / dx**2 + ky2), -np.ones(n - 2) / dx**2],
            [-1, 0, 1],
        )
        A = diags(1 / (d + damping)) @ (diags(2 * d) - C0**2 * laplace)
        B = diags(-(d - damping) / (d + damping))
        companion = bmat([[A, B], [eye(n - 1), None]], format="csc", dtype=complex)
        target = np.exp(-1j * truth * c.dt)
        z = eigs(companion, k=1, sigma=target, return_eigenvectors=False, tol=1e-11)[0]
        omega = 1j * np.log(z) / c.dt
    return dict(
        n=n,
        contrast=contrast,
        sigma_e=sigma,
        mapping=mode,
        samples=samples,
        frequency_error=abs(omega.real / truth.real - 1),
        decay_error=abs(omega.imag / truth.imag - 1) if sigma else None,
        analytic_frequency_hz=truth.real / (2 * np.pi),
        analytic_decay_per_second=-truth.imag,
        diagnostics=c.averaging,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for contrast, sigma in [(12, 0), (30, 0), (12, 0.1)]:
        for n in [32, 64, 128, 256, 512]:
            for mode, samples in [("point", 8), ("sampled", 8), ("sampled", 16), ("sampled", 32)]:
                rows.append(measure(n, contrast, sigma, mode, samples))
        print("Finished", contrast, sigma, flush=True)
    (args.output / "controlled.json").write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, (contrast, sigma) in zip(axes, [(12, 0), (30, 0), (12, 0.1)]):
        for mode, samples in [("point", 8), ("sampled", 8), ("sampled", 16), ("sampled", 32)]:
            rr = [
                r
                for r in rows
                if (r["contrast"], r["sigma_e"], r["mapping"], r["samples"])
                == (contrast, sigma, mode, samples)
            ]
            ax.loglog(
                [r["n"] for r in rr],
                [100 * r["frequency_error"] for r in rr],
                "o-",
                label="Point" if mode == "point" else f"{samples}×{samples}",
            )
        ax.set(
            title=f"εr={contrast}, σe={sigma} S/m",
            xlabel="Cells per axis",
            ylabel="Frequency error [%]",
        )
        ax.set_xticks([32, 64, 128, 256, 512], labels=["32", "64", "128", "256", "512"])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Production sampled-material averaging vs analytical slab-cavity modes")
    fig.savefig(args.output / "controlled.png", dpi=160)
    plt.close(fig)
    print(json.dumps([r for r in rows if r["n"] == 512], indent=2))


if __name__ == "__main__":
    main()
