"""PyTorch CUDA backend for TMz scattering and streaming surface DFT.

The numerical equations match ``solver.simulate``. Geometry averaging remains a
one-time CPU preprocessing step; fields, CPML state, analytic source evaluation,
and DFT accumulation stay on the selected Torch device during time stepping.
PEC intersection geometry remains one-time CPU preprocessing. Cut-edge gradients,
total-field constraints, and projected cell-enlargement transfers run on device.
"""

from time import perf_counter

import numpy as np

from .conformal import CutCellPEC
from .constants import C0, EPS0, MU0
from .geometry import PEC, average_materials
from .observables import Contour, SurfaceDFT
from .solver import SimulationResult, _pml_profile


def _torch_module():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("The CUDA backend requires PyTorch") from error
    return torch


def _pulse(torch, time, source):
    shifted = time - source.delay
    return torch.exp(-((shifted / source.width) ** 2)) * torch.cos(
        2 * np.pi * source.frequency * shifted
    )


def _contour_fields(torch, ez, hx, hy, contour):
    i0, i1, j0, j1 = contour.i0, contour.i1, contour.j0, contour.j1
    electric, magnetic = [], []
    for i, sign in ((i0, -1), (i1, 1)):
        electric.append((ez[i, j0:j1] + ez[i, j0 + 1 : j1 + 1]) / 2)
        left = contour.grid.x[i] - contour.grid.x[i - 1]
        right = contour.grid.x[i + 1] - contour.grid.x[i]
        h = (right * hy[i - 1, :] + left * hy[i, :]) / (left + right)
        magnetic.append(sign * (h[j0:j1] + h[j0 + 1 : j1 + 1]) / 2)
    for j, sign in ((j0, 1), (j1, -1)):
        electric.append((ez[i0:i1, j] + ez[i0 + 1 : i1 + 1, j]) / 2)
        left = contour.grid.y[j] - contour.grid.y[j - 1]
        right = contour.grid.y[j + 1] - contour.grid.y[j]
        h = (right * hx[:, j - 1] + left * hx[:, j]) / (left + right)
        magnetic.append(sign * (h[i0:i1] + h[i0 + 1 : i1 + 1]) / 2)
    return torch.cat(electric), torch.cat(magnetic)


def _as_tensor(torch, value, *, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


class _TorchPEC:
    """Device representation of a CPU-constructed ``CutCellPEC`` operator."""

    def __init__(self, torch, cut, source, *, device, dtype):
        self.torch, self.cut, self.source = torch, cut, source

        def real(value):
            return torch.as_tensor(value, device=device, dtype=dtype)

        def index(value):
            return torch.as_tensor(value, device=device, dtype=torch.long)

        self.inverse_lengths = tuple(real(value) for value in cut.inverse_lengths)
        self.inside = torch.as_tensor(cut.inside, device=device, dtype=torch.bool)
        retardation = source.retardation(cut.grid.x[:, None], cut.grid.y[None, :])
        self.inside_delay = real(retardation[cut.inside])
        self.boundaries = []
        for boundary in cut.boundaries:
            if boundary is None:
                self.boundaries.append(None)
                continue
            i, j, ei, ej, x, y, sign = boundary
            delay = source.retardation(x, y)
            self.boundaries.append(
                (index(i), index(j), index(ei), index(ej), real(delay), real(sign))
            )

        enlargement = cut.enlargement
        self.enlargement = enlargement is not None
        if enlargement is None:
            return
        self.masters = index(enlargement.masters)
        self.slaves = index(enlargement.slaves)
        self.roots = index(enlargement.roots)
        self.slave_owners = index(enlargement.slave_owners)
        self.slave_weights = real(enlargement.slave_weights)
        self.master_mass = real(enlargement.master_mass)
        self.slave_mass_weight = real(enlargement.slave_mass_weight)
        self.mass = real(enlargement.mass)
        self.slave_delay = real(source.retardation(*enlargement.slave_positions))
        self.root_delay = real(source.retardation(*enlargement.root_positions))
        self.previous_g = self._incident_offset(0.0)

    def _incident_offset(self, time):
        return self.slave_weights * _pulse(
            self.torch, time - self.root_delay, self.source
        ) - _pulse(self.torch, time - self.slave_delay, self.source)

    def initialize(self, ez):
        self.impose_inside(ez, 0.0)
        if self.enlargement:
            flat = ez.ravel()
            flat[self.slaves] = self.slave_weights * flat[self.roots] + self.previous_g

    def impose_inside(self, ez, time):
        ez[self.inside] = -_pulse(self.torch, time - self.inside_delay, self.source)

    def gradients(self, ez, time):
        gradients = [
            self.torch.diff(ez, dim=axis) * inverse
            for axis, inverse in enumerate(self.inverse_lengths)
        ]
        for gradient, inverse, boundary in zip(gradients, self.inverse_lengths, self.boundaries):
            if boundary is None:
                continue
            i, j, ei, ej, delay, sign = boundary
            incident = _pulse(self.torch, time - delay, self.source)
            gradient[i, j] = sign * (ez[ei, ej] + incident) * inverse[i, j]
        return gradients

    def advance(self, ez, increment, time):
        g = self._incident_offset(time)
        delta = increment.ravel()
        numerator = self.master_mass * delta[self.masters]
        transferred = self.slave_mass_weight * (delta[self.slaves] - (g - self.previous_g))
        numerator = numerator.index_add(0, self.slave_owners, transferred)
        flat = ez.ravel()
        flat[self.masters] += numerator / self.mass
        flat[self.slaves] = self.slave_weights * flat[self.roots] + g
        self.previous_g = g


def simulate_cuda(
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
    device="cuda",
    dtype="float64",
    phase_reanchor_interval=2048,
    check_interval=256,
    pec_mode="conformal",
):
    """Run the scattered-field solver on a Torch device.

    ``device="cpu"`` exists for deterministic backend-equivalence tests. CUDA
    production checks should use float64 first; float32 requires separate error
    qualification. DFT phase recurrence is reset to the analytic phase at the
    declared interval to bound accumulated rotation error.
    """
    torch = _torch_module()
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    real_dtype = getattr(torch, dtype)
    complex_dtype = torch.complex64 if dtype == "float32" else torch.complex128
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch in this process")
    if (
        isinstance(phase_reanchor_interval, bool)
        or int(phase_reanchor_interval) != phase_reanchor_interval
        or phase_reanchor_interval < 1
        or isinstance(check_interval, bool)
        or int(check_interval) != check_interval
        or check_interval < 1
    ):
        raise ValueError("Phase reanchor and check intervals must be positive integers")
    objects = tuple(objects)
    started = perf_counter()
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
    pec_objects = tuple(obj for obj in objects if isinstance(obj.material, PEC))
    if pec_objects and len(pec_objects) != len(objects):
        raise ValueError("Mixed PEC/dielectric coupling is not yet qualified in this prototype")
    cut_pec = CutCellPEC(grid, pec_objects, pec_mode) if pec_objects else None
    grid_dt = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
    stable_dt = min(grid_dt, cut_pec.stable_time_step()) if cut_pec else grid_dt
    dt = safety * stable_dt
    nt = int(np.ceil(duration / dt))
    if nt > max_steps:
        raise ValueError(f"Simulation requires {nt} steps, exceeding {max_steps}")
    frequencies = np.asarray(frequencies, dtype=float)
    if (
        frequencies.ndim != 1
        or len(frequencies) == 0
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
    ):
        raise ValueError("Positive finite observation frequencies required")
    if frequencies.max() * dt >= 0.5 or source.frequency * dt >= 0.5:
        raise ValueError("Source or requested DFT frequencies exceed temporal Nyquist limit")
    if monitor_bounds is None:
        monitor_bounds = (0.25 * lx, 0.75 * lx, 0.25 * ly, 0.75 * ly)
    contour = Contour(grid, monitor_bounds)
    i0, i1, j0, j1 = contour.i0, contour.i1, contour.j0, contour.j1
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
    eps, sigma = average_materials(grid, () if cut_pec else objects, samples)
    retardation = source.retardation(grid.x[:, None], grid.y[None, :])
    if np.max(source.envelope(-retardation)) > 1e-8:
        raise ValueError("Source starts before the simulation: increase pulse delay")

    def tensor(value):
        return _as_tensor(torch, value, device=device, dtype=real_dtype)

    dx_t, dy_t = tensor(dx), tensor(dy)
    dualx = tensor((dx[:-1] + dx[1:]) / 2)
    dualy = tensor((dy[:-1] + dy[1:]) / 2)
    eps_t, sigma_t = tensor(eps), tensor(sigma)
    loss = sigma_t * dt / (2 * EPS0 * eps_t)
    ca, cb = (1 - loss) / (1 + loss), dt / (EPS0 * eps_t * (1 + loss))
    contrast, conduction = (1 - 1 / eps_t) / (1 + loss), loss / (1 + loss)
    active = (contrast != 0) | (conduction != 0)
    retardation_t = tensor(retardation)
    delay = retardation_t[active]
    previous_incident = _pulse(torch, -delay, source)
    ez = torch.zeros(grid.shape, device=device, dtype=real_dtype)
    hx = torch.zeros((len(grid.x), len(grid.y) - 1), device=device, dtype=real_dtype)
    hy = torch.zeros((len(grid.x) - 1, len(grid.y)), device=device, dtype=real_dtype)
    phx, phy = torch.zeros_like(hx), torch.zeros_like(hy)
    pex, pey = torch.zeros_like(ez), torch.zeros_like(ez)
    torch_pec = (
        _TorchPEC(torch, cut_pec, source, device=device, dtype=real_dtype) if cut_pec else None
    )
    if torch_pec:
        torch_pec.initialize(ez)
    xc, yc = grid.centers
    ex, ey = [
        tuple(tensor(value) for value in _pml_profile(axis, length, thick, dt))
        for axis, length, thick in ((grid.x, lx, thickness[0]), (grid.y, ly, thickness[1]))
    ]
    hpx, hpy = [
        tuple(tensor(value) for value in _pml_profile(axis, length, thick, dt))
        for axis, length, thick in ((xc, lx, thickness[0]), (yc, ly, thickness[1]))
    ]
    omega = tensor(2 * np.pi * frequencies)
    phase_step = torch.exp(1j * omega * dt).to(complex_dtype)
    phase_e = phase_step.clone()
    phase_h = torch.exp(1j * omega * (0.5 * dt)).to(complex_dtype)
    electric_times = torch.arange(1, nt + 1, device=device, dtype=real_dtype) * dt
    incident_at_origin = _pulse(torch, electric_times, source)
    npoints = len(contour.weights)
    electric_dft = torch.zeros((len(frequencies), npoints), device=device, dtype=complex_dtype)
    magnetic_dft = torch.zeros_like(electric_dft)
    incident_dft = torch.zeros(len(frequencies), device=device, dtype=complex_dtype)
    incident_l1 = torch.zeros((), device=device, dtype=real_dtype)
    peak = torch.zeros((), device=device, dtype=real_dtype)
    tail_peak = torch.zeros((), device=device, dtype=real_dtype)
    interior = np.s_[1:-1, 1:-1]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_started = perf_counter()
    with torch.inference_mode():
        for n in range(nt):
            if torch_pec:
                gx, gy = torch_pec.gradients(ez, n * dt)
            else:
                gy = torch.diff(ez, dim=1) / dy_t[None, :]
                gx = torch.diff(ez, dim=0) / dx_t[:, None]
            phx.mul_(hpy[1][None, :]).add_(hpy[2][None, :] * gy)
            phy.mul_(hpx[1][:, None]).add_(hpx[2][:, None] * gx)
            hx.sub_(dt / MU0 * (hpy[0][None, :] * gy + phx))
            hy.add_(dt / MU0 * (hpx[0][:, None] * gx + phy))
            gx = torch.diff(hy, dim=0)[:, 1:-1] / dualx[:, None]
            gy = torch.diff(hx, dim=1)[1:-1, :] / dualy[None, :]
            pex[interior].mul_(ex[1][1:-1, None]).add_(ex[2][1:-1, None] * gx)
            pey[interior].mul_(ey[1][None, 1:-1]).add_(ey[2][None, 1:-1] * gy)
            curl = ex[0][1:-1, None] * gx + pex[interior] - ey[0][None, 1:-1] * gy - pey[interior]
            te = electric_times[n]
            if torch_pec and torch_pec.enlargement:
                increment = torch.zeros_like(ez)
                increment[interior] = cb[interior] * curl
                torch_pec.advance(ez, increment, te)
            else:
                ez[interior].mul_(ca[interior]).add_(cb[interior] * curl)
            incident = _pulse(torch, te - delay, source)
            ez[active] -= contrast[active] * (incident - previous_incident) + conduction[active] * (
                incident + previous_incident
            )
            previous_incident = incident
            if torch_pec:
                torch_pec.impose_inside(ez, te)
            e_surface, h_surface = _contour_fields(torch, ez, hx, hy, contour)
            electric_dft.add_(dt * phase_e[:, None] * e_surface[None, :])
            magnetic_dft.add_(dt * phase_h[:, None] * h_surface[None, :])
            incident_origin = incident_at_origin[n]
            incident_dft.add_(dt * phase_e * incident_origin)
            incident_l1.add_(dt * abs(incident_origin))
            maximum = torch.max(torch.abs(ez))
            peak = torch.maximum(peak, maximum)
            if n >= int(0.9 * nt):
                tail_peak = torch.maximum(tail_peak, maximum)
            if (n + 1) % check_interval == 0:
                checked = float(maximum.item())
                if not np.isfinite(checked) or checked > 1e8:
                    raise RuntimeError(
                        "Scattered field became nonfinite or exceeded the amplitude limit"
                    )
            phase_e *= phase_step
            phase_h *= phase_step
            if (n + 1) % phase_reanchor_interval == 0:
                phase_e = torch.exp(1j * omega * ((n + 2) * dt)).to(complex_dtype)
                phase_h = torch.exp(1j * omega * ((n + 1.5) * dt)).to(complex_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_maximum = float(torch.max(torch.abs(ez)).item())
    if not np.isfinite(final_maximum) or final_maximum > 1e8:
        raise RuntimeError("Scattered field became nonfinite or exceeded the amplitude limit")
    step_seconds = perf_counter() - step_started
    monitor = SurfaceDFT(contour, frequencies)
    monitor.electric = electric_dft.cpu().numpy()
    monitor.tangential_h = magnetic_dft.cpu().numpy()
    monitor.incident = incident_dft.cpu().numpy()
    monitor.incident_l1 = float(incident_l1.item())
    fields = dict(Ez=ez.cpu().numpy(), Hx=hx.cpu().numpy(), Hy=hy.cpu().numpy())
    peak_value, tail_value = float(peak.item()), float(tail_peak.item())
    diagnostics = dict(
        **grid.diagnostics(),
        backend="torch_" + device.type,
        device=str(device),
        dtype=dtype,
        formulation="analytic_incident_scattered_field",
        dt=dt,
        Nt=nt,
        simulated_time=nt * dt,
        requested_time=duration,
        cell_updates=(len(grid.x) - 1) * (len(grid.y) - 1) * nt,
        wall_seconds=perf_counter() - started,
        stepping_seconds=step_seconds,
        peak_scattered_e=peak_value,
        tail_peak_over_global_peak=tail_value / max(peak_value, 1e-30),
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
        dft_phase_method="recurrence_with_analytic_reanchor",
        dft_phase_reanchor_interval=int(phase_reanchor_interval),
        finite_check_interval=int(check_interval),
        host_full_field_transfers=1,
        host_scalar_checks=int(np.ceil(nt / check_interval)) + 4,
    )
    return SimulationResult(monitor, fields, diagnostics)
