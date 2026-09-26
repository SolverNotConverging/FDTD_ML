"""Pre/post processing for the Cython-bound, persistent CUDA TMz solver.

The FDTD time loop, CPML, PEC cut-edge treatment, source, field checks, and
surface/point DFT all execute in one cooperative CUDA kernel. NumPy is used
only before launch and after the final device-to-host copy.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .conformal import CutCellPEC
from .constants import C0, EPS0, MU0
from .geometry import average_materials, split_material_objects
from .observables import Contour, SurfaceDFT, field_point_stencil
from .solver import SimulationResult, _pml_profile

INPUT_NAMES = (
    "ca",
    "cb",
    "contrast",
    "conduction",
    "delay",
    "active",
    "inside",
    "inverse_x",
    "inverse_y",
    "boundary_exterior_x",
    "boundary_exterior_y",
    "boundary_delay_x",
    "boundary_delay_y",
    "boundary_sign_x",
    "boundary_sign_y",
    "inverse_dual_x",
    "inverse_dual_y",
    "ex_k",
    "ex_b",
    "ex_a",
    "ey_k",
    "ey_b",
    "ey_a",
    "hpx_k",
    "hpx_b",
    "hpx_a",
    "hpy_k",
    "hpy_b",
    "hpy_a",
    "x",
    "y",
    "phase_e_real",
    "phase_e_imag",
    "phase_h_real",
    "phase_h_imag",
    "incident_origin",
    "point_i",
    "point_j",
    "point_wx",
    "point_wy",
    "point_delay",
    "masters",
    "slaves",
    "roots",
    "slave_owners",
    "slave_weights",
    "master_mass",
    "slave_mass_weight",
    "aggregate_mass",
    "root_delay",
    "slave_delay",
)


def _cuda_device_index(device):
    value = str(device)
    if value == "cuda":
        return 0
    if value.startswith("cuda:") and value[5:].isdigit():
        return int(value[5:])
    raise ValueError("Compiled FDTD device must be cuda or cuda:<index>")


def _pec_edge_inputs(grid, cut, source):
    nx, ny = grid.shape
    if cut is None:
        inverse_x = np.broadcast_to(1 / np.diff(grid.x)[:, None], (nx - 1, ny)).copy()
        inverse_y = np.broadcast_to(1 / np.diff(grid.y)[None, :], (nx, ny - 1)).copy()
    else:
        inverse_x, inverse_y = cut.inverse_lengths
    ext_x = np.full((nx - 1, ny), -1, dtype=np.int32)
    ext_y = np.full((nx, ny - 1), -1, dtype=np.int32)
    delay_x = np.zeros_like(ext_x, dtype=np.float64)
    delay_y = np.zeros_like(ext_y, dtype=np.float64)
    sign_x = np.zeros_like(ext_x, dtype=np.float64)
    sign_y = np.zeros_like(ext_y, dtype=np.float64)
    if cut is not None:
        for exterior, delay, sign, boundary in (
            (ext_x, delay_x, sign_x, cut.boundaries[0]),
            (ext_y, delay_y, sign_y, cut.boundaries[1]),
        ):
            if boundary is None:
                continue
            i, j, ei, ej, x, y, boundary_sign = boundary
            exterior[i, j] = ei * ny + ej
            delay[i, j] = source.retardation(x, y)
            sign[i, j] = boundary_sign
    return (inverse_x, inverse_y, ext_x, ext_y, delay_x, delay_y, sign_x, sign_y)


def simulate_compiled_cuda(
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
    device="cuda:0",
    dtype="float64",
    phase_reanchor_interval=2048,
    check_interval=256,
    pec_mode="conformal",
    field_sample_points=None,
):
    """Run all TMz time steps in one compiled CUDA kernel with no step transfers."""
    try:
        from . import _cuda_fdtd
    except ImportError as error:
        raise RuntimeError(
            "Compiled CUDA solver is missing; run `.venv/bin/python setup_cuda.py build_ext --inplace`"
        ) from error
    root = Path(__file__).parent
    build_record = root / "_cuda_fdtd_build.json"
    if not build_record.is_file():
        raise RuntimeError("Compiled CUDA build record is missing; rebuild with setup_cuda.py")
    recorded = json.loads(build_record.read_text())
    current = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("_cuda_fdtd.pyx", "cuda_fdtd_kernels.h", "cuda_fdtd_kernels.cu")
    }
    if recorded.get("source_sha256") != current:
        raise RuntimeError("Compiled CUDA kernel is stale; rebuild with setup_cuda.py")
    if dtype != "float64":
        raise ValueError("Compiled CUDA FDTD currently requires float64")
    if not isinstance(check_interval, int) or check_interval < 1:
        raise ValueError("Check interval must be a positive integer")
    if not isinstance(phase_reanchor_interval, int) or phase_reanchor_interval < 1:
        raise ValueError("Phase reanchor interval must be a positive integer")
    started = perf_counter()
    device_index = _cuda_device_index(device)
    objects = tuple(objects)
    nx, ny = grid.shape
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
    cut = CutCellPEC(grid, pec_objects, pec_mode) if pec_objects else None
    grid_dt = 1 / (C0 * np.sqrt(dx.min() ** -2 + dy.min() ** -2))
    stable_dt = min(grid_dt, cut.stable_time_step()) if cut else grid_dt
    dt = safety * stable_dt
    nt = int(np.ceil(duration / dt))
    if nt > max_steps:
        raise ValueError(f"Simulation requires {nt} steps, exceeding {max_steps}")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or not len(frequencies)
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

    eps, sigma = average_materials(grid, dielectric_objects, samples)
    retardation = source.retardation(grid.x[:, None], grid.y[None, :])
    if np.max(source.envelope(-retardation)) > 1e-8:
        raise ValueError("Source starts before the simulation: increase pulse delay")
    loss = sigma * dt / (2 * EPS0 * eps)
    ca = (1 - loss) / (1 + loss)
    cb = dt / (EPS0 * eps * (1 + loss))
    contrast = (1 - 1 / eps) / (1 + loss)
    conduction = loss / (1 + loss)
    active = ((contrast != 0) | (conduction != 0)).astype(np.uint8)
    inside = cut.inside.astype(np.uint8) if cut else np.zeros(grid.shape, dtype=np.uint8)
    active[inside.astype(bool)] = 0
    enlargement = cut.enlargement if cut else None
    if enlargement is not None:
        transfer_nodes = np.r_[enlargement.slaves, enlargement.roots]
        if active.ravel()[transfer_nodes].any():
            raise ValueError(
                "PEC enlargement transfer stencil intersects dielectric material; "
                "use conformal mode or separate the objects"
            )
        enlarged_inputs = (
            np.asarray(enlargement.masters, dtype=np.int32),
            np.asarray(enlargement.slaves, dtype=np.int32),
            np.asarray(enlargement.roots, dtype=np.int32),
            np.asarray(enlargement.slave_owners, dtype=np.int32),
            enlargement.slave_weights,
            enlargement.master_mass,
            enlargement.slave_mass_weight,
            enlargement.mass,
            source.retardation(*enlargement.root_positions),
            source.retardation(*enlargement.slave_positions),
        )
    else:
        enlarged_inputs = tuple(
            np.empty(0, dtype=np.int32 if index < 4 else np.float64) for index in range(10)
        )
    edge_inputs = _pec_edge_inputs(grid, cut, source)
    dualx, dualy = (dx[:-1] + dx[1:]) / 2, (dy[:-1] + dy[1:]) / 2
    xc, yc = grid.centers
    ex = _pml_profile(grid.x, lx, thickness[0], dt)
    ey = _pml_profile(grid.y, ly, thickness[1], dt)
    hpx = _pml_profile(xc, lx, thickness[0], dt)
    hpy = _pml_profile(yc, ly, thickness[1], dt)
    te = (np.arange(nt, dtype=np.float64) + 1) * dt
    th = (np.arange(nt, dtype=np.float64) + 0.5) * dt
    electric_phase = 2 * np.pi * te[:, None] * frequencies[None, :]
    magnetic_phase = 2 * np.pi * th[:, None] * frequencies[None, :]
    incident_origin = source.pulse(te)
    point_stencil = (
        field_point_stencil(grid, field_sample_points) if field_sample_points is not None else None
    )
    if point_stencil is None:
        point_i = point_j = np.empty(0, dtype=np.int32)
        point_wx = point_wy = point_delay = np.empty(0, dtype=np.float64)
        npoints = 0
    else:
        point_i, point_j, point_wx, point_wy = point_stencil
        point_i = np.asarray(point_i, dtype=np.int32)
        point_j = np.asarray(point_j, dtype=np.int32)
        points = np.asarray(field_sample_points, dtype=float)
        point_delay = source.retardation(points[:, 0], points[:, 1])
        npoints = len(points)
    raw_inputs = (
        ca,
        cb,
        contrast,
        conduction,
        retardation,
        active,
        inside,
        *edge_inputs,
        1 / dualx,
        1 / dualy,
        *ex,
        *ey,
        *hpx,
        *hpy,
        grid.x,
        grid.y,
        np.cos(electric_phase),
        np.sin(electric_phase),
        np.cos(magnetic_phase),
        np.sin(magnetic_phase),
        incident_origin,
        point_i,
        point_j,
        point_wx,
        point_wy,
        point_delay,
        *enlarged_inputs,
    )
    if len(raw_inputs) != len(INPUT_NAMES):
        raise RuntimeError("Compiled CUDA input schema is inconsistent")
    inputs = [np.ascontiguousarray(array).ravel() for array in raw_inputs]
    nf = len(frequencies)
    ncontour = len(contour.weights)
    outputs = [
        np.empty(nx * ny, dtype=np.float64),
        np.empty(nx * (ny - 1), dtype=np.float64),
        np.empty((nx - 1) * ny, dtype=np.float64),
        np.empty((nf, ncontour, 2), dtype=np.float64),
        np.empty((nf, ncontour, 2), dtype=np.float64),
        np.empty((nf, 2), dtype=np.float64),
        np.empty((nf, npoints, 2), dtype=np.float64),
        np.empty((nf, npoints, 2), dtype=np.float64),
        np.empty(nx * ny, dtype=np.float64),
        np.empty(nx * ny, dtype=np.float64),
        np.empty(1, dtype=np.int32),
    ]
    stats = _cuda_fdtd.run(
        {
            "nx": nx,
            "ny": ny,
            "nt": nt,
            "nf": nf,
            "npoints": npoints,
            "nmasters": len(enlargement.masters) if enlargement else 0,
            "nslaves": len(enlargement.slaves) if enlargement else 0,
            "i0": i0,
            "i1": i1,
            "j0": j0,
            "j1": j1,
            "device_index": device_index,
            "dt": dt,
            "mu0": MU0,
            "source_frequency": source.frequency,
            "source_delay": source.delay,
            "source_width": source.width,
        },
        inputs,
        outputs,
    )
    if outputs[10][0]:
        raise RuntimeError("Scattered field became nonfinite or exceeded the amplitude limit")
    monitor = SurfaceDFT(contour, frequencies)
    monitor.electric = outputs[3].view(np.complex128).reshape(nf, ncontour)
    monitor.tangential_h = outputs[4].view(np.complex128).reshape(nf, ncontour)
    monitor.incident = outputs[5].view(np.complex128).reshape(nf)
    monitor.incident_l1 = float(dt * np.sum(np.abs(incident_origin)))
    fields = {
        "Ez": outputs[0].reshape(nx, ny),
        "Hx": outputs[1].reshape(nx, ny - 1),
        "Hy": outputs[2].reshape(nx - 1, ny),
    }
    if npoints:
        fields.update(
            Ez_point_scattered_dft=outputs[6].view(np.complex128).reshape(nf, npoints),
            Ez_point_incident_dft=outputs[7].view(np.complex128).reshape(nf, npoints),
            field_sample_points=np.asarray(field_sample_points, dtype=float),
        )
    peak = float(np.max(outputs[8]))
    tail = float(np.max(outputs[9]))
    diagnostics = dict(
        **grid.diagnostics(),
        backend="compiled_cuda",
        device=f"cuda:{device_index}",
        dtype=dtype,
        formulation="analytic_incident_scattered_field",
        dt=dt,
        Nt=nt,
        simulated_time=nt * dt,
        requested_time=duration,
        cell_updates=(nx - 1) * (ny - 1) * nt,
        wall_seconds=perf_counter() - started,
        stepping_seconds=stats["kernel_seconds"],
        peak_scattered_e=peak,
        tail_peak_over_global_peak=tail / max(peak, 1e-30),
        incident_tail_at_origin=float(abs(source.pulse(nt * dt))),
        averaging_samples_per_axis=samples,
        monitor_bounds=list(contour.bounds),
        incident_angle_radians=source.angle,
        incident_phase_origin=list(source.origin),
        far_field_origin=[0.0, 0.0],
        fourier_convention="integral f(t) exp(+i omega t) dt",
        pml_thickness=thickness.tolist(),
        pec_mode=pec_mode if cut else None,
        pec_node_count=int(cut.inside.sum()) if cut else 0,
        pec_boundary_edge_count=cut.boundary_edge_count if cut else 0,
        pec_subcell_edge_count=cut.subcell_edge_count if cut else 0,
        pec_minimum_open_fraction=cut.minimum_fraction if cut else 1.0,
        dt_fraction_of_grid_cfl=stable_dt / grid_dt,
        pec_enlarged_nodes=len(enlargement.slaves) if enlargement else 0,
        pec_coordinate_tolerance_m=cut.coordinate_tolerance if cut else 0.0,
        dft_phase_method="precomputed_analytic_phase",
        dft_phase_reanchor_interval=0,
        finite_check_interval=0,
        host_full_field_transfers=1,
        host_scalar_checks=0,
        host_transfers_during_steps=0,
        host_to_device_bytes=stats["host_to_device_bytes"],
        device_to_host_bytes=stats["device_to_host_bytes"],
        peak_cuda_memory_bytes=stats["allocated_bytes"],
        cooperative_blocks=stats["cooperative_blocks"],
        threads_per_block=stats["threads_per_block"],
    )
    return SimulationResult(monitor, fields, diagnostics)
