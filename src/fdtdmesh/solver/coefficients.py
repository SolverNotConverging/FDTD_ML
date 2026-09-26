"""Vacuum TMz coefficients and certified conservative spatial-operator bound."""

from dataclasses import dataclass

import numpy as np

from ..constants import C0, EPS0, MU0
from ..pml import build_cpml
from .conformal import build_conformal


@dataclass
class Coefficients:
    pec: np.ndarray
    hx: object
    hy: object
    ex: np.ndarray
    ey: np.ndarray
    ih: np.ndarray
    cpml: np.ndarray
    dt: float
    dt_cartesian: float
    dt_bound: float
    spectral_bound: float


def build_coefficients(scene, mesh, *, pml=None, dt=None, safety=0.9, dtype="float64"):
    if dtype not in ("float32", "float64") or not 0 < safety < 1:
        raise ValueError("Require float32/float64 and 0<safety<1")
    if mesh.x[-1] != scene.Lx or mesh.y[-1] != scene.Ly:
        raise ValueError("Mesh and physical domain differ")
    pec, hx, hy, bound = build_conformal(scene, mesh)
    dx, dy = np.diff(mesh.x), np.diff(mesh.y)
    ux = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    uy = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    cartesian = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
    stable = min(cartesian, 2 / (C0 * np.sqrt(bound)))
    dt = safety * stable if dt is None else float(dt)
    if not np.isfinite(dt) or not 0 < dt < stable:
        raise ValueError(f"dt must be positive and below conformal bound {stable}")
    for op in (hx, hy):
        op.diagonal = np.ascontiguousarray(op.diagonal * dt / MU0, dtype=dtype)
        op.coupling = np.ascontiguousarray(op.coupling * dt / MU0, dtype=dtype)
    return Coefficients(
        pec.astype(np.uint8),
        hx,
        hy,
        np.asarray(dt / EPS0 / ux, dtype=dtype),
        np.asarray(dt / EPS0 / uy, dtype=dtype),
        np.asarray(dt / MU0 / dx, dtype=dtype),
        build_cpml(scene, mesh, dt, dtype, pml),
        dt,
        cartesian,
        stable,
        bound,
    )
