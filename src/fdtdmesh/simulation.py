"""GPU-resident conformal PEC scattering with device-controlled convergence."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .constants import C0
from .mesh import MeshInfeasibleError
from .scattering import box_indices, grid_index, make_contour
from .solver.coefficients import build_coefficients


@dataclass(frozen=True)
class Convergence:
    """DFT-bin settling policy. max_steps is a safety cap, not a time window."""

    max_steps: int = 200_000
    check_interval: int = 2048
    stable_checks: int = 3
    rtol: float = 1e-5
    atol: float = 1e-8
    field_tol: float = 1e-5

    def __post_init__(self):
        for n in (self.max_steps, self.check_interval, self.stable_checks):
            if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n < 1:
                raise ValueError("Convergence counts must be positive integers")
        if self.max_steps > 10_000_000 or self.max_steps < self.check_interval:
            raise ValueError("Invalid timestep resource cap")
        if (
            not np.isfinite([self.rtol, self.atol, self.field_tol]).all()
            or min(self.rtol, self.atol, self.field_tol) <= 0
        ):
            raise ValueError("Convergence tolerances must be finite and positive")


@dataclass(frozen=True)
class ScatteringCase:
    scene: object
    frequency: float
    tfsf_box: tuple
    contour_box: tuple
    pml: object
    source_x: float
    origin: tuple
    pulse_width_periods: float = 1.5
    pulse_delay_periods: float = 6.0
    frequencies: tuple = ()
    angles: np.ndarray = field(default_factory=lambda: np.deg2rad(np.arange(360)))
    convergence: Convergence = field(default_factory=Convergence)
    pulse_end: float | None = None

    def __post_init__(self):
        scalars = [self.frequency, self.pulse_width_periods, self.pulse_delay_periods]
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError("Frequency and pulse durations must be positive")
        if self.pulse_end is not None and (not np.isfinite(self.pulse_end) or self.pulse_end <= 0):
            raise ValueError("pulse_end must be finite and positive")
        freq = np.asarray(self.frequencies if len(self.frequencies) else (self.frequency,), float)
        if freq.ndim != 1 or not np.isfinite(freq).all() or np.any(freq <= 0):
            raise ValueError("Requested frequencies must be finite and positive")
        # Refuse bins with negligible source excitation instead of declaring zero-response convergence.
        if np.any(
            np.exp(-((np.pi * self.pulse_width_periods * (freq / self.frequency - 1)) ** 2)) < 1e-4
        ):
            raise ValueError("A requested DFT bin lies outside the usable pulse bandwidth")


@dataclass
class ScatteringResult:
    frequencies: np.ndarray
    angles: np.ndarray
    amplitude: np.ndarray
    width: np.ndarray
    diagnostics: dict
    history: np.ndarray
    mesh: object
    debug: dict = field(default_factory=dict)
    bin_history: np.ndarray | None = None

    @property
    def converged(self):
        return self.diagnostics["status"] == "converged"

    def save(self, path):
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frequencies=self.frequencies,
            angles=self.angles,
            amplitude=self.amplitude,
            width=self.width,
            history=self.history,
            x=self.mesh.x,
            y=self.mesh.y,
            diagnostics=json.dumps(self.diagnostics, allow_nan=False),
        )


class ConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(
            f"FDTD stopped with {result.diagnostics['status']}; results are not qualified"
        )


class MeshClearanceError(MeshInfeasibleError):
    """A candidate grid lacks cells required to separate scattering regions."""


def prepare(case, mesh, *, dtype="float64", safety=0.9, dt=None):
    scene = case.scene
    case.pml.validate_scene(scene)
    case.pml.validate_mesh(mesh)
    box = box_indices(mesh, case.tfsf_box)
    contour = make_contour(mesh, case.contour_box)
    a, b, c, d = box
    ca, cb, cc, cd = box_indices(mesh, case.contour_box)
    if not ca + 1 < a < b < cb - 1 or not cc + 1 < c < d < cd - 1:
        raise MeshClearanceError("Contour must enclose TFSF with at least two cells of clearance")
    px, py = case.pml.x.cells, case.pml.y.cells
    if not px < ca < cb < mesh.Nx - px or not py < cc < cd < mesh.Ny - py:
        raise MeshClearanceError("Contour and H interpolation must be strictly outside PML")
    material = scene.material_bounds()
    for x0, x1, y0, y1 in () if material is None else (material,):
        if not (
            mesh.x[a + 2] < x0 < x1 < mesh.x[b - 2] and mesh.y[c + 2] < y0 < y1 < mesh.y[d - 2]
        ):
            raise MeshClearanceError("Geometry and enlarged cells need clearance inside TFSF")
    source = grid_index(mesh.x, case.source_x)
    if not px < source < ca:
        raise MeshClearanceError("Auxiliary incident source must lie between left PML and contour")
    ix = grid_index(mesh.x, case.origin[0])
    if not a < ix < b or not case.tfsf_box[2] < case.origin[1] < case.tfsf_box[3]:
        raise ValueError("Phase origin must be inside TFSF and x-anchored")
    coeff = build_coefficients(
        scene,
        mesh,
        pml=case.pml,
        dtype=dtype,
        safety=safety,
        dt=dt,
    )
    # Do not test convergence before the pulse and two domain transits have passed.
    source_end = (
        (case.pulse_delay_periods + 6 * case.pulse_width_periods) / case.frequency
        if case.pulse_end is None
        else case.pulse_end
    )
    earliest = source_end + 2 * np.hypot(scene.Lx, scene.Ly) / C0
    minimum = int(np.ceil(earliest / coeff.dt))
    policy = case.convergence
    if policy.max_steps <= minimum:
        raise ValueError("Safety cap ends before the source/propagation convergence guard")
    options = dict(
        max_steps=policy.max_steps,
        check_interval=policy.check_interval,
        stable_checks=policy.stable_checks,
        min_steps=minimum,
        rtol=policy.rtol,
        atol=policy.atol,
        field_tol=policy.field_tol,
        frequency=case.frequency,
        frequencies=case.frequencies if len(case.frequencies) else (case.frequency,),
        angles=case.angles,
        pulse_width=case.pulse_width_periods,
        pulse_delay=case.pulse_delay_periods,
        source_scale=2 * C0 * coeff.dt / ((mesh.x[source + 1] - mesh.x[source - 1]) / 2),
        origin=case.origin,
        origin_index=ix,
        pulse_end=case.pulse_end if case.pulse_end is not None else 1e100,
    )
    return coeff, box, source, contour, options


def run_scattering(
    case,
    mesh,
    *,
    dtype="float64",
    safety=0.9,
    dt=None,
    progress=None,
    diagnostic_download=False,
    require_converged=True,
    _prepared=None,
):
    """Run native CUDA FDTD -> current DFT -> convergence -> NF2FF -> final download.

    The test-only diagnostic download occurs AFTER GPU NF2FF and is off by default.
    Progress callbacks can be slow or skipped without holding up the GPU graph.
    """
    from .solver.tmz import cuda_backend

    runtime = cuda_backend()
    start = perf_counter()
    coeff, box, source, contour, options = (
        _prepared
        if _prepared is not None
        else prepare(case, mesh, dtype=dtype, safety=safety, dt=dt)
    )
    options = dict(options)
    setup = perf_counter() - start
    options["debug"] = diagnostic_download
    amplitude, width, history, stats, debug = runtime.run(
        coeff, (mesh.Nx, mesh.Ny), box, source, contour, options, progress=progress
    )
    bin_history = stats.pop("bin_history")
    if stats["status"] != "nonfinite" and (
        not np.isfinite(amplitude).all() or not np.isfinite(width).all()
    ):
        stats["status"] = "nonfinite"
    stats.update(
        mesh.diagnostics(),
        dt=float(coeff.dt),
        dt_cartesian=float(coeff.dt_cartesian),
        dt_bound=float(coeff.dt_bound),
        spectral_bound=coeff.spectral_bound,
        simulated_time=stats["Nt"] * coeff.dt,
        frequency=case.frequency,
        cell_updates=mesh.Nx * mesh.Ny * stats["Nt"],
        setup_seconds=setup,
        wall_seconds=perf_counter() - start,
        enlarged_pairs=len(coeff.hx.pairs) + len(coeff.hy.pairs),
        cut_faces=int(
            np.count_nonzero(
                (coeff.hx.length > 0) & (coeff.hx.length < np.diff(mesh.y)[None, :] * (1 - 1e-10))
            )
            + np.count_nonzero(
                (coeff.hy.length > 0) & (coeff.hy.length < np.diff(mesh.x)[:, None] * (1 - 1e-10))
            )
        ),
        method="tmz-conformal-ect-v1",
        dtype=dtype,
        configuration=dict(
            scene=case.scene.as_dict(),
            pml=asdict(case.pml),
            tfsf_box=case.tfsf_box,
            contour_box=case.contour_box,
            source_x=case.source_x,
            origin=case.origin,
            pulse_width_periods=case.pulse_width_periods,
            pulse_delay_periods=case.pulse_delay_periods,
            pulse_end=case.pulse_end,
            convergence=asdict(case.convergence),
            minimum_steps=options["min_steps"],
        ),
    )
    result = ScatteringResult(
        np.asarray(options["frequencies"]),
        np.asarray(case.angles),
        amplitude,
        width,
        stats,
        history,
        mesh,
        debug,
        bin_history,
    )
    if require_converged and not result.converged:
        raise ConvergenceError(result)
    return result
