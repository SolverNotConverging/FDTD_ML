"""Explicit NumPy scattered-field TMz solver on nonuniform Yee grids.

mu_r=1, sigma_h=0. The incident field is analytic; material contrast drives
scattered fields volumetrically. No TFSF boundary or point-source anchors.
This is a numerical reference implementation, not a CUDA production backend.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .conformal import CutCellPEC
from .constants import C0, EPS0, MU0, Z0
from .geometry import average_materials, split_material_objects
from .observables import Contour, SurfaceDFT


def _pml_profile(coordinates, length, thickness, dt):
    u = np.maximum(
        (thickness - coordinates) / thickness, (coordinates - (length - thickness)) / thickness
    ).clip(0, 1)
    sigma = -4 * np.log(1e-8) / (2 * Z0 * thickness) * u**3
    kappa = 1 + 2 * u**3
    alpha = 0.05 * (1 - u) * (u > 0)
    exponent = -(sigma / kappa + alpha) * dt / EPS0
    b = np.exp(exponent)
    a = np.zeros_like(u)
    denominator = sigma * kappa + alpha * kappa * kappa
    np.divide(sigma * np.expm1(exponent), denominator, out=a, where=denominator > 0)
    return 1 / kappa, b, a


@dataclass
class SimulationResult:
    monitor: SurfaceDFT
    fields: dict
    diagnostics: dict


def simulate(
    grid,
    objects,
    source,
    *,
    frequencies,
    duration,
    pml_thickness,
    monitor_bounds=None,
    samples=8,
    safety=0.9,
    max_steps=200_000,
    pec_mode="conformal",
):
    started = perf_counter()
    objects = tuple(objects)
    lx, ly = grid.x[-1], grid.y[-1]
    thickness = np.broadcast_to(np.asarray(pml_thickness, dtype=float), (2,))
    if (
        not np.isfinite(thickness).all()
        or np.any(thickness <= 0)
        or np.any(2 * thickness >= [lx, ly])
    ):
        raise ValueError("Positive PML thickness must leave an interior")
    if not np.isfinite(safety) or not 0 < safety < 1 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Positive duration and CFL safety between 0 and 1 required")
    dx, dy = np.diff(grid.x), np.diff(grid.y)
    pec_objects, dielectric_objects = split_material_objects(objects)
    cut_pec = CutCellPEC(grid, pec_objects, pec_mode) if pec_objects else None
    grid_dt = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
    stable_dt = min(grid_dt, cut_pec.stable_time_step()) if cut_pec else grid_dt
    dt = safety * stable_dt
    nt = int(np.ceil(duration / dt))
    if nt > max_steps:
        raise ValueError(f"Simulation requires {nt} steps, exceeding {max_steps}")
    if monitor_bounds is None:
        monitor_bounds = (0.25 * lx, 0.75 * lx, 0.25 * ly, 0.75 * ly)
    contour = Contour(grid, monitor_bounds)
    i0, i1, j0, j1 = contour.i0, contour.i1, contour.j0, contour.j1
    # Entire interpolation stencil must remain in homogeneous, non-PML background.
    if (
        grid.x[i0 - 1] <= thickness[0]
        or grid.x[i1 + 1] >= lx - thickness[0]
        or grid.y[j0 - 1] <= thickness[1]
        or grid.y[j1 + 1] >= ly - thickness[1]
    ):
        raise ValueError("Monitor interpolation stencil enters PML")
    for obj in objects:
        a, b, c, d = obj.bounds
        if not (
            grid.x[i0 + 1] < a < b < grid.x[i1 - 1] and grid.y[j0 + 1] < c < d < grid.y[j1 - 1]
        ):
            raise ValueError(
                "Every scatterer must lie strictly inside the monitor with a vacuum buffer"
            )
    eps, sigma = average_materials(grid, dielectric_objects, samples)
    loss = sigma * dt / (2 * EPS0 * eps)
    ca, cb = (1 - loss) / (1 + loss), dt / (EPS0 * eps * (1 + loss))
    contrast, conduction = (1 - 1 / eps) / (1 + loss), loss / (1 + loss)
    retardation = source.retardation(grid.x[:, None], grid.y[None, :])
    if np.max(source.envelope(-retardation)) > 1e-8:
        raise ValueError("Source starts before the simulation: increase pulse delay")
    # Restrict source evaluation to material nodes. Empty space remains exactly zero.
    active = (contrast != 0) | (conduction != 0)
    if cut_pec:
        active[cut_pec.inside] = False
        if cut_pec.enlargement is not None:
            aggregate = cut_pec.enlargement
            transfer_nodes = np.r_[aggregate.slaves, aggregate.roots]
            if active.ravel()[transfer_nodes].any():
                raise ValueError(
                    "PEC enlargement transfer stencil intersects dielectric material; "
                    "use conformal mode or separate the objects"
                )
    delay = retardation[active]
    previous_incident = source.pulse(-delay)
    dualx, dualy = (dx[:-1] + dx[1:]) / 2, (dy[:-1] + dy[1:]) / 2
    ez = np.zeros(grid.shape)
    if cut_pec:
        ez[cut_pec.inside] = -source.pulse(-retardation[cut_pec.inside])
        if cut_pec.enlargement is not None:
            cut_pec.enlargement.initialize(ez, source)
    hx = np.zeros((len(grid.x), len(grid.y) - 1))
    hy = np.zeros((len(grid.x) - 1, len(grid.y)))
    phx, phy, pex, pey = np.zeros_like(hx), np.zeros_like(hy), np.zeros_like(ez), np.zeros_like(ez)
    xc, yc = grid.centers
    ex, ey = [
        _pml_profile(a, length, thick, dt)
        for a, length, thick in ((grid.x, lx, thickness[0]), (grid.y, ly, thickness[1]))
    ]
    hpx, hpy = [
        _pml_profile(a, length, thick, dt)
        for a, length, thick in ((xc, lx, thickness[0]), (yc, ly, thickness[1]))
    ]
    monitor = SurfaceDFT(contour, frequencies)
    if monitor.frequencies.max() * dt >= 0.5:
        raise ValueError("Requested DFT frequencies exceed temporal Nyquist limit")
    if source.frequency * dt >= 0.5:
        raise ValueError("Source carrier exceeds temporal Nyquist limit")
    peak, tail_peak = 0.0, 0.0
    for n in range(nt):
        if cut_pec:
            gx, gy = cut_pec.gradients(ez, source, n * dt)
        else:
            gy, gx = np.diff(ez, axis=1) / dy[None, :], np.diff(ez, axis=0) / dx[:, None]
        phx = hpy[1][None, :] * phx + hpy[2][None, :] * gy
        phy = hpx[1][:, None] * phy + hpx[2][:, None] * gx
        hx -= dt / MU0 * (hpy[0][None, :] * gy + phx)
        hy += dt / MU0 * (hpx[0][:, None] * gx + phy)
        gx = np.diff(hy, axis=0)[:, 1:-1] / dualx[:, None]
        gy = np.diff(hx, axis=1)[1:-1, :] / dualy[None, :]
        interior = np.s_[1:-1, 1:-1]
        pex[interior] = ex[1][1:-1, None] * pex[interior] + ex[2][1:-1, None] * gx
        pey[interior] = ey[1][None, 1:-1] * pey[interior] + ey[2][None, 1:-1] * gy
        curl = ex[0][1:-1, None] * gx + pex[interior] - ey[0][None, 1:-1] * gy - pey[interior]
        te, th = (n + 1) * dt, (n + 0.5) * dt
        if cut_pec and cut_pec.enlargement is not None:
            increment = np.zeros_like(ez)
            increment[interior] = (ca[interior] - 1) * ez[interior] + cb[interior] * curl
            cut_pec.enlargement.advance(ez, increment, source, te)
        else:
            ez[interior] = ca[interior] * ez[interior] + cb[interior] * curl
        incident = source.pulse(te - delay)
        ez[active] -= contrast[active] * (incident - previous_incident) + conduction[active] * (
            incident + previous_incident
        )
        previous_incident = incident
        if cut_pec:
            ez[cut_pec.inside] = -source.pulse(te - retardation[cut_pec.inside])
        monitor.accumulate(
            ez, hx, hy, electric_time=te, magnetic_time=th, dt=dt, incident=source.pulse(te)
        )
        maximum = float(np.max(abs(ez)))
        peak = max(peak, maximum)
        if n >= int(0.9 * nt):
            tail_peak = max(tail_peak, maximum)
        if not np.isfinite(maximum) or maximum > 1e8:
            raise RuntimeError("Scattered field became nonfinite or exceeded the amplitude limit")
    return SimulationResult(
        monitor,
        dict(Ez=ez, Hx=hx, Hy=hy),
        dict(
            **grid.diagnostics(),
            backend="numpy_reference",
            formulation="analytic_incident_scattered_field",
            dt=dt,
            Nt=nt,
            simulated_time=nt * dt,
            requested_time=duration,
            cell_updates=(len(grid.x) - 1) * (len(grid.y) - 1) * nt,
            wall_seconds=perf_counter() - started,
            peak_scattered_e=peak,
            tail_peak_over_global_peak=tail_peak / max(peak, 1e-30),
            incident_tail_at_origin=float(abs(source.pulse(nt * dt))),
            averaging_samples_per_axis=samples,
            monitor_bounds=list(contour.bounds),
            incident_angle_radians=source.angle,
            incident_phase_origin=list(source.origin),
            far_field_origin=[0.0, 0.0],
            fourier_convention="integral f(t) exp(+i omega t) dt",
            pml_thickness=thickness.tolist(),
            pec_mode=pec_mode if cut_pec else None,
            pec_node_count=int(cut_pec.inside.sum()) if cut_pec else 0,
            pec_boundary_edge_count=cut_pec.boundary_edge_count if cut_pec else 0,
            pec_subcell_edge_count=cut_pec.subcell_edge_count if cut_pec else 0,
            pec_minimum_open_fraction=cut_pec.minimum_fraction if cut_pec else 1.0,
            dt_fraction_of_grid_cfl=stable_dt / grid_dt,
            pec_enlarged_nodes=len(cut_pec.enlargement.slaves)
            if cut_pec and cut_pec.enlargement is not None
            else 0,
            pec_coordinate_tolerance_m=cut_pec.coordinate_tolerance if cut_pec else 0.0,
        ),
    )
