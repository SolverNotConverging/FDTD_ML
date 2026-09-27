"""Define simulation -> add exact geometry -> apply mesh -> solve -> save/plot."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .constants import C0
from .geometry import Geometry
from .mesh import AxisCollar, Mesh
from .pml import PML
from .pulse import GaussianPulse
from .simulation import Convergence, ConvergenceError, ScatteringCase, prepare, run_scattering
from .strategies import cnn_density, generate_mesh, mesh_id

DFTConvergence = Convergence


@dataclass(frozen=True)
class SolverSettings:
    """DFT frequency count/vector (Hz), device stopping, field precision and CFL safety."""

    dft_bins: int | tuple = 21
    stop: Convergence = field(default_factory=Convergence)
    precision: str = "float64"
    courant: float = 0.9

    def __post_init__(self):
        bins = self.dft_bins
        if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
            if bins < 1:
                raise ValueError("dft_bins must be positive")
            object.__setattr__(self, "dft_bins", int(bins))
        else:
            a = np.asarray(bins, float)
            if (
                a.ndim != 1
                or not a.size
                or not np.isfinite(a).all()
                or np.any(a <= 0)
                or np.any(np.diff(a) <= 0)
            ):
                raise ValueError("Explicit DFT bins must be increasing positive frequencies in Hz")
            object.__setattr__(self, "dft_bins", tuple(a.tolist()))
        if (
            not isinstance(self.stop, Convergence)
            or self.precision not in ("float32", "float64")
            or not 0 < self.courant < 1
        ):
            raise ValueError("Invalid stopping policy, precision or courant multiplier")

    def frequencies(self, fmin, fmax):
        b = self.dft_bins
        f = (
            np.array([(fmin + fmax) / 2])
            if b == 1
            else np.linspace(fmin, fmax, b)
            if isinstance(b, int)
            else np.array(b)
        )
        if np.any(f < fmin) or np.any(f > fmax):
            raise ValueError("DFT frequencies must lie within fmin/fmax")
        return f


@dataclass(frozen=True)
class ScatteringLayout:
    tfsf_box: tuple
    contour_box: tuple
    source_x: float
    origin: tuple

    def __post_init__(self):
        for name, count in (("tfsf_box", 4), ("contour_box", 4), ("origin", 2)):
            a = tuple(float(v) for v in getattr(self, name))
            if len(a) != count or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, a)
        if not np.isfinite(self.source_x):
            raise ValueError("Invalid source_x")


class Simulation:
    """SI-unit TMz PEC scattering with a +x broadband plane wave.

    Define continuous geometry, apply a mesh, then call :meth:`solve`. Geometry
    edits invalidate the prepared grid. FDTD, current DFT, stopping and NF2FF are
    native CUDA; construction, geometry, meshing and archive loading need no GPU.
    """

    def __init__(self, size, fmin, fmax, *, settings=None, angles=None, pml=None, layout=None):
        self._geometry = Geometry(tuple(size))
        self._pulse = GaussianPulse(float(fmin), float(fmax))
        self._settings = settings or SolverSettings()
        self._settings.frequencies(fmin, fmax)
        angles = np.deg2rad(np.arange(360)) if angles is None else np.asarray(angles, float)
        if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
            raise ValueError("angles must be a finite nonempty vector in radians")
        self._angles = tuple(angles.tolist())
        lam = self.wavelength
        collar = AxisCollar(12, 0.5 * lam)
        self._pml = pml or PML(collar, collar)
        lx, ly = self.size
        px, py = self.pml.x.thickness, self.pml.y.thickness
        self._layout = layout or ScatteringLayout(
            (px + lam, lx - px - lam, py + lam, ly - py - lam),
            (px + 0.5 * lam, lx - px - 0.5 * lam, py + 0.5 * lam, ly - py - 0.5 * lam),
            px + 0.25 * lam,
            (lx / 2, ly / 2),
        )
        a, b, c, d = self.layout.tfsf_box
        if not 0 < a < b < lx or not 0 < c < d < ly:
            raise ValueError(
                "Domain is too small for the scattering layout; enlarge size or supply a layout"
            )
        self._mesh = self._prepared = self._case = self._result = None
        self._mesh_seconds = 0.0

    @property
    def geometry(self):
        return self._geometry

    @property
    def size(self):
        return self.geometry.size

    @property
    def source(self):
        return self._pulse

    @property
    def fmin(self):
        return self.source.fmin

    @property
    def fmax(self):
        return self.source.fmax

    @property
    def wavelength(self):
        return C0 / self.source.frequency

    @property
    def settings(self):
        return self._settings

    @property
    def layout(self):
        return self._layout

    @property
    def pml(self):
        return self._pml

    @property
    def mesh(self):
        return self._mesh

    @property
    def result(self):
        return self._result

    @property
    def frequencies(self):
        return self.settings.frequencies(self.fmin, self.fmax)

    def _invalidate(self):
        self._mesh = self._prepared = self._case = self._result = None

    def _add(self, kind, parameters, material, name):
        g, handle = self.geometry.added(kind, parameters, material, name)
        self._geometry = g
        self._invalidate()
        return handle

    def add_circle(self, center, radius, material="PEC", name=None):
        """Overlay an analytic circle in metres; return its removable shape handle."""
        return self._add("circle", (*center, radius), material, name)

    def add_rectangle(self, x, y, material="PEC", name=None):
        """Overlay rectangle ranges (x0,x1), (y0,y1) in metres; return a handle."""
        return self._add("rectangle", (*x, *y), material, name)

    def add_polygon(self, vertices, material="PEC", name=None):
        """Overlay a simple polygon with vertices in metres; return a handle."""
        return self._add("polygon", vertices, material, name)

    def remove_geometry(self, handle):
        self._geometry = self.geometry.removed(handle)
        self._invalidate()

    def configure_solver(self, **changes):
        """Replace SolverSettings fields and invalidate any prepared mesh/result."""
        from dataclasses import replace

        settings = replace(self.settings, **changes)
        settings.frequencies(self.fmin, self.fmax)
        self._settings = settings
        self._invalidate()

    def _make_case(self):
        p, layout = self.source, self.layout
        return ScatteringCase(
            self.geometry.to_scene(),
            p.frequency,
            layout.tfsf_box,
            layout.contour_box,
            self.pml,
            layout.source_x,
            layout.origin,
            pulse_width_periods=p.tau * p.frequency,
            pulse_delay_periods=p.delay * p.frequency,
            frequencies=tuple(self.frequencies),
            angles=np.array(self._angles),
            convergence=self.settings.stop,
            pulse_end=p.duration,
        )

    def apply_mesh(
        self,
        strategy="uniform",
        *,
        cells=None,
        density=None,
        checkpoint=None,
        strict=False,
        constraints=None,
        time_limit=30.0,
    ):
        """Prepare a grid and conformal enlarged-cell coefficients; return the Mesh.

        strategy: uniform, deterministic, density, cnn, or an explicit Mesh.
        cells includes PML. strict=True requires an exactly uniform grid.
        Checkpoint bundles and density vectors are described in docs/solver_api.md.
        A failed proposal leaves the previously applied mesh intact.
        """
        start = perf_counter()
        metadata = {}
        if isinstance(strategy, Mesh):
            if (
                cells is not None
                or density is not None
                or checkpoint is not None
                or strict
                or constraints is not None
            ):
                raise ValueError("An explicit Mesh cannot be combined with generator options")
            candidate = Mesh(strategy.x, strategy.y, dict(strategy.metadata))
        else:
            if cells is None:
                raise ValueError("Specify cells=(Nx, Ny), including fixed PML collars")
            actual_strategy = strategy
            if strategy == "cnn":
                if checkpoint is None or density is not None or strict:
                    raise ValueError("cnn requires checkpoint and does not accept density/strict")
                density, metadata = cnn_density(self.geometry, self.fmin, self.fmax, checkpoint)
                actual_strategy, checkpoint = "density", None
            candidate = generate_mesh(
                self.geometry,
                self.layout,
                self.pml,
                cells,
                actual_strategy,
                density=density,
                checkpoint=checkpoint,
                strict=strict,
                constraints=constraints,
                time_limit=time_limit,
            )
            candidate = Mesh(
                candidate.x, candidate.y, {**candidate.metadata, **metadata, "strategy": strategy}
            )
        case = self._make_case()
        prepared = prepare(
            case, candidate, dtype=self.settings.precision, safety=self.settings.courant
        )
        # Validate the actual sampled source, not just its continuous design spectrum.
        dt = prepared[0].dt
        if self.source.significant_frequency * dt >= 0.5:
            raise ValueError("Mesh timestep undersamples the significant Gaussian spectrum")
        n = int(np.ceil(self.source.duration / dt))
        if n > 2_000_000:
            raise ValueError(
                "Pulse sampling exceeds two million samples; check band width and mesh"
            )
        times = np.arange(1, n + 1) * dt
        wave = self.source(times)
        spectrum = np.array(
            [np.sum(wave * np.exp(-2j * np.pi * f * times)) * dt for f in self.frequencies]
        )
        center = abs(np.sum(wave * np.exp(-2j * np.pi * self.source.frequency * times)) * dt)
        if center < 1e-20 or np.any(abs(spectrum) / center < 1e-3):
            raise ValueError("A requested bin is under-excited by the sampled Gaussian")
        self._mesh, self._case, self._prepared, self._result = candidate, case, prepared, None
        self._mesh_seconds = perf_counter() - start
        return candidate

    @property
    def discretization(self):
        if self._prepared is None:
            raise RuntimeError("Call apply_mesh before inspecting Yee discretization")
        # Return an independent inspection snapshot so callers cannot corrupt a cached solve.
        from copy import deepcopy

        return deepcopy(self._prepared[0])

    def summary(self):
        data = dict(
            size=self.size,
            units="SI",
            geometry_id=self.geometry.geometry_id,
            fmin=self.fmin,
            fmax=self.fmax,
            frequencies=self.frequencies.tolist(),
            pulse=asdict(self.source),
            source_end=self.source.duration,
            significant_frequency=self.source.significant_frequency,
            layout=asdict(self.layout),
            pml=asdict(self.pml),
        )
        if self.mesh is not None:
            data.update(
                self.mesh.diagnostics(),
                mesh_id=mesh_id(self.mesh),
                dt=self._prepared[0].dt,
                enlarged_pairs=len(self._prepared[0].hx.pairs) + len(self._prepared[0].hy.pairs),
                is_uniform=all(
                    np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-10, atol=0)
                    for a in (self.mesh.x, self.mesh.y)
                ),
                min_ppw_significant=C0
                / self.source.significant_frequency
                / max(np.diff(self.mesh.x).max(), np.diff(self.mesh.y).max()),
            )
        return data

    def solve(self, *, progress=False, diagnostic_download=False, require_converged=True):
        """Run the prepared problem on CUDA and return an immutable Result snapshot.

        progress may be True or an asynchronous presentation callback. Failed
        convergence raises ConvergenceError with its unqualified .result unless
        require_converged=False. diagnostic_download opts into final field/current
        transfers after GPU NF2FF; ordinary results contain only far fields/history.
        """
        from .result import Result

        if self._prepared is None:
            raise RuntimeError("Call apply_mesh after defining or changing the simulation")
        if progress is True:

            def callback(p):
                print(
                    f"step {p['step']}: DFT ratio {p['dft_error_ratio']:.3g}; residual {p['residual']:.3g}",
                    flush=True,
                )
        elif progress is False or progress is None:
            callback = None
        elif callable(progress):
            callback = progress
        else:
            raise ValueError("progress must be True, False, or callable")
        self._result = None
        raw = run_scattering(
            self._case,
            self.mesh,
            dtype=self.settings.precision,
            safety=self.settings.courant,
            progress=callback,
            diagnostic_download=diagnostic_download,
            require_converged=False,
            _prepared=self._prepared,
        )
        raw.diagnostics.update(
            geometry_id=self.geometry.geometry_id,
            mesh_id=mesh_id(self.mesh),
            meshing_seconds=self._mesh_seconds,
            source=asdict(self.source),
        )
        result = Result.from_run(raw, self.geometry, self.configuration())
        if not result.converged and require_converged:
            raise ConvergenceError(result)
        self._result = result
        return result

    def configuration(self):
        return dict(
            size=self.size,
            fmin=self.fmin,
            fmax=self.fmax,
            settings=asdict(self.settings),
            angles=self._angles,
            pml=asdict(self.pml),
            layout=asdict(self.layout),
        )

    def plot_geometry(self, *, mesh=False, units="m", ax=None):
        from .plotting import geometry_plot

        if mesh and self.mesh is None:
            raise RuntimeError("Call apply_mesh before plotting grid lines")
        return geometry_plot(
            self.geometry, self.mesh if mesh else None, self.configuration(), units, ax
        )

    def plot_mesh(self, *, units="m", ax=None):
        from .plotting import mesh_plot

        return mesh_plot(self.mesh, self.wavelength, units, ax)

    def plot_discretization(self, *, units="m", ax=None):
        from .plotting import discretization_plot

        return discretization_plot(
            self.geometry, self.mesh, self.discretization, self.configuration(), units, ax
        )

    def plot_source(self, ax=None):
        from .plotting import source_plot

        return source_plot(self.source, self.frequencies, ax)

    def save(self, path):
        from .storage import save_simulation

        save_simulation(self, path)

    @classmethod
    def load(cls, path):
        from .storage import load_simulation

        return load_simulation(path)
