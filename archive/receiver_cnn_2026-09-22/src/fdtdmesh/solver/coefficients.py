from dataclasses import dataclass, field

import numpy as np

from ..constants import C0, EPS0, MU0
from ..pml import build_cpml
from .averaging import average_electric


@dataclass
class Coefficients:
    ca: np.ndarray
    cbx: np.ndarray
    cby: np.ndarray
    chx: np.ndarray
    chy: np.ndarray
    ahx: np.ndarray
    ahy: np.ndarray
    pec: np.ndarray
    dt: float
    dt_cfl: float
    cpml: np.ndarray
    current_scale: np.ndarray
    averaging: dict = field(default_factory=lambda: {"mode": "point"})


def build_coefficients(scene, mesh, *, dt=None, safety=0.95, dtype="float32"):
    if dtype not in ("float32", "float64"):
        raise ValueError("Only float32 and float64 fields are supported")
    if not np.isfinite(safety) or not 0 < safety < 1:
        raise ValueError("CFL safety must lie strictly between zero and one")
    dx, dy = np.diff(mesh.x), np.diff(mesh.y)
    eps, _, sigma, pec = scene.sample(mesh.x, mesh.y)
    averaging = {"mode": "point"}
    if getattr(scene, "material_averaging", "point") == "sampled":
        eps, sigma, averaging = average_electric(
            scene,
            mesh,
            eps,
            sigma,
            pec,
            samples=scene.averaging_samples,
            max_samples=scene.averaging_max_samples,
            tolerance=scene.averaging_tolerance,
        )
    _, muhx, _, shx, _ = scene.sample(mesh.x, (mesh.y[:-1] + mesh.y[1:]) / 2, magnetic_loss=True)
    _, muhy, _, shy, _ = scene.sample((mesh.x[:-1] + mesh.x[1:]) / 2, mesh.y, magnetic_loss=True)
    # Global separate minima also bound heterogeneous epsilon/mu on staggered sites.
    cmax = C0 / np.sqrt(eps.min() * min(muhx.min(), muhy.min()))
    cfl = 1 / (cmax * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
    timestep = safety * cfl if dt is None else float(dt)
    if not np.isfinite(timestep) or timestep <= 0 or timestep >= cfl:
        raise ValueError(f"dt must be positive and strictly below CFL bound {cfl:.6g}")
    loss = sigma * timestep / (2 * EPS0 * eps)
    ca = (1 - loss) / (1 + loss)
    cb = timestep / (EPS0 * eps) / (1 + loss)
    loss_hx = shx * timestep / (2 * MU0 * muhx)
    loss_hy = shy * timestep / (2 * MU0 * muhy)
    dualx = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    dualy = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    pec[[0, -1], :] = True
    pec[:, [0, -1]] = True
    arrays = (
        ca,
        cb / dualx[:, None],
        cb / dualy[None, :],
        timestep / (MU0 * muhx * dy[None, :]) / (1 + loss_hx),
        timestep / (MU0 * muhy * dx[:, None]) / (1 + loss_hy),
        (1 - loss_hx) / (1 + loss_hx),
        (1 - loss_hy) / (1 + loss_hy),
    )
    arrays = [np.ascontiguousarray(a, dtype=dtype) for a in arrays]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Coefficients overflow the requested field precision")
    return Coefficients(
        *arrays,
        np.ascontiguousarray(pec, dtype=np.uint8),
        timestep,
        cfl,
        build_cpml(scene, mesh, timestep, dtype),
        np.ascontiguousarray(cb, dtype=dtype),
        averaging,
    )
