"""Define simulation -> add exact geometry -> apply mesh -> solve -> save/plot."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .constants import C0
from .domain import DomainPolicy
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
    exterior_cells: tuple | None = None
    scatterer_margin_cells: int | None = None

    def __post_init__(self):
        for name, count in (("tfsf_box", 4), ("contour_box", 4), ("origin", 2)):
            a = tuple(float(v) for v in getattr(self, name))
            if len(a) != count or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, a)
        if not np.isfinite(self.source_x):
            raise ValueError("Invalid source_x")
        if self.exterior_cells is not None:
            from .mesh import cell_count

            counts = tuple(cell_count(n) for n in self.exterior_cells)
            if len(counts) != 2 or counts[0] < 2 or counts[1] < 2:
                raise ValueError("exterior_cells needs at least two cells in each gap")
            object.__setattr__(self, "exterior_cells", counts)
        if self.scatterer_margin_cells is not None:
            from .mesh import cell_count

            if cell_count(self.scatterer_margin_cells) < 3:
                raise ValueError("scatterer_margin_cells must be at least three")
            if self.exterior_cells is None:
                raise ValueError("Scatterer cell margins require exterior cell allocation")


class Simulation:
    """SI-unit TMz PEC scattering with a +x broadband plane wave.

    Define continuous geometry, apply a mesh, then call :meth:`solve`. Geometry
    edits invalidate the prepared grid. FDTD, current DFT, stopping and NF2FF are
    native CUDA; construction, geometry, meshing and archive loading need no GPU.
    """

    def __init__(
        self,
        size=None,
        fmin=None,
        fmax=None,
        *,
        settings=None,
        angles=None,
        pml=None,
        layout=None,
        domain=None,
    ):
        if fmin is None or fmax is None:
            raise ValueError("Specify fmin and fmax in Hz")
        self._automatic_domain = size is None
        if domain is not None and (size is not None or not isinstance(domain, DomainPolicy)):
            raise ValueError("DomainPolicy requires size=None")
        if self._automatic_domain and (pml is not None or layout is not None):
            raise ValueError("Use DomainPolicy for automatic-domain PML and margins")
        self._domain_policy = (domain or DomainPolicy()) if self._automatic_domain else None
        self._resolved_geometry = None
        self._coordinate_offset = (0.0, 0.0)
        self._geometry = Geometry(None if size is None else tuple(size))
        self._pulse = GaussianPulse(float(fmin), float(fmax))
        self._settings = settings or SolverSettings()
        self._settings.frequencies(fmin, fmax)
        angles = np.deg2rad(np.arange(360)) if angles is None else np.asarray(angles, float)
        if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
            raise ValueError("angles must be a finite nonempty vector in radians")
        self._angles = tuple(angles.tolist())
        self._mesh = self._prepared = self._case = self._result = None
        self._mesh_seconds = 0.0
        if self._automatic_domain:
            self._pml = self._layout = None
            return
        lam = self.wavelength
        collar = AxisCollar(12, 0.5 * lam)
        self._pml = pml or PML(collar, collar)
        lx, ly = self.size
        px, py = self.pml.x.thickness, self.pml.y.thickness
        hx = px / self.pml.x.cells if self.pml.x.cells else lam / 24
        hy = py / self.pml.y.cells if self.pml.y.cells else lam / 24
        self._layout = layout or ScatteringLayout(
            (px + 10 * hx, lx - px - 10 * hx, py + 10 * hy, ly - py - 10 * hy),
            (px + 6 * hx, lx - px - 6 * hx, py + 6 * hy, ly - py - 6 * hy),
            px + 3 * hx,
            (lx / 2, ly / 2),
            exterior_cells=(6, 4),
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
        return self.computational_geometry.size

    def _resolve_domain(self):
        if self._automatic_domain and self._resolved_geometry is None:
            values = self._domain_policy.resolve(self.geometry, self.fmax)
            self._resolved_geometry, self._pml, self._layout, self._coordinate_offset = values

    @property
    def computational_geometry(self):
        """Exact geometry translated into the zero-based solver domain."""
        self._resolve_domain()
        return self._resolved_geometry if self._automatic_domain else self.geometry

    @property
    def coordinate_offset(self):
        """Solver coordinates = input coordinates + this recorded translation."""
        self._resolve_domain()
        return self._coordinate_offset

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
        self._resolve_domain()
        return self._layout

    @property
    def pml(self):
        self._resolve_domain()
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
        if self._automatic_domain:
            self._resolved_geometry = None

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

    def add_ellipse(self, center, radii, angle=0.0, material="PEC", name=None):
        """Add an analytic ellipse: semiaxes in metres, rotation in radians."""
        return self._add("ellipse", (*center, *radii, angle), material, name)

    def set_geometry(self, geometry):
        """Install an immutable geometry snapshot in this domain; invalidate mesh."""
        if not isinstance(geometry, Geometry) or (
            not self._automatic_domain and geometry.size != self.size
        ):
            raise ValueError("Geometry must match this simulation domain")
        self._geometry = geometry
        self._invalidate()

    def fit_domain(self, *, scatterer_margin_cells=5, exterior_cells=(6, 4)):
        """Fit the domain to final PEC bounds with fixed cell margins; invalidate mesh.

        Translate the entire exact recipe and phase origin together. Physical
        gap widths use the adjacent PML cell widths; PML thickness is unchanged.
        Call after adding geometry and before applying a mesh.
        """
        from .mesh import cell_count

        if self._automatic_domain:
            from dataclasses import replace

            self._domain_policy = replace(
                self._domain_policy,
                scatterer_to_tfsf=scatterer_margin_cells,
                pml_to_contour=exterior_cells[0],
                contour_to_tfsf=exterior_cells[1],
            )
            self._invalidate()
            self._resolve_domain()
            return self

        margin = cell_count(scatterer_margin_cells)
        gaps = tuple(cell_count(n) for n in exterior_cells)
        if margin < 3 or len(gaps) != 2 or min(gaps) < 2:
            raise ValueError("Need >=3 scatterer-margin cells and >=2 cells in each exterior gap")
        bounds = self.geometry.bounds
        if bounds is None:
            raise ValueError("Cannot fit a domain without PEC material")
        h = np.array(
            [
                p.thickness / p.cells if p.cells else self.wavelength / 24
                for p in (self.pml.x, self.pml.y)
            ]
        )
        thickness = np.array([self.pml.x.thickness, self.pml.y.thickness])
        padding = thickness + (sum(gaps) + margin) * h
        low, high = np.array(bounds)[[0, 2]], np.array(bounds)[[1, 3]]
        offset = padding - low
        size = high - low + 2 * padding
        geometry = self.geometry.translated(offset, size=tuple(size))
        t = thickness + sum(gaps) * h
        c = thickness + gaps[0] * h
        layout = ScatteringLayout(
            (t[0], size[0] - t[0], t[1], size[1] - t[1]),
            (c[0], size[0] - c[0], c[1], size[1] - c[1]),
            thickness[0] + (gaps[0] // 2) * h[0],
            tuple(np.asarray(self.layout.origin) + offset),
            exterior_cells=gaps,
            scatterer_margin_cells=margin,
        )
        self._geometry, self._layout = geometry, layout
        self._invalidate()
        return self

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
            self.computational_geometry.to_scene(),
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
        anchors=None,
        anchor_assignment="joint",
        time_limit=30.0,
        target_spacing=None,
        max_cells=(512, 512),
        max_passes=20,
    ):
        """Prepare a grid and conformal enlarged-cell coefficients; return the Mesh.

        strategy: uniform, deterministic, density, cnn, geometry_aware, or an explicit Mesh.
        cells includes PML. strict=True requires an exactly uniform grid.
        Checkpoint bundles and density vectors are described in docs/solver_api.md.
        A failed proposal leaves the previously applied mesh intact.
        geometry_aware selects counts using target_spacing, max_cells, max_passes,
        and the construction time_limit. See docs/geometry_aware_meshing.md.
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
                or anchors is not None
                or anchor_assignment != "joint"
                or target_spacing is not None
                or max_cells != (512, 512)
                or max_passes != 20
            ):
                raise ValueError("An explicit Mesh cannot be combined with generator options")
            candidate = Mesh(strategy.x, strategy.y, dict(strategy.metadata))
        elif strategy == "geometry_aware":
            from .geometry_mesher import geometry_aware_mesh

            if (
                cells is not None
                or density is not None
                or checkpoint is not None
                or strict
                or anchors is not None
                or anchor_assignment != "joint"
            ):
                raise ValueError(
                    "geometry_aware chooses cell counts; use target_spacing and max_cells"
                )
            candidate = geometry_aware_mesh(
                self.computational_geometry,
                self.layout,
                self.pml,
                target_spacing=C0 / self.fmax / 32 if target_spacing is None else target_spacing,
                constraints=constraints,
                max_cells=max_cells,
                max_passes=max_passes,
                time_limit=time_limit,
            )
        else:
            if target_spacing is not None or max_cells != (512, 512) or max_passes != 20:
                raise ValueError("target_spacing/max_cells/max_passes require geometry_aware")
            if cells is None:
                raise ValueError("Specify cells=(Nx, Ny), including fixed PML collars")
            actual_strategy = strategy
            if strategy == "cnn":
                if checkpoint is None or density is not None or strict:
                    raise ValueError("cnn requires checkpoint and does not accept density/strict")
                density, metadata = cnn_density(
                    self.computational_geometry, self.fmin, self.fmax, checkpoint
                )
                actual_strategy, checkpoint = "density", None
            candidate = generate_mesh(
                self.computational_geometry,
                self.layout,
                self.pml,
                cells,
                actual_strategy,
                density=density,
                checkpoint=checkpoint,
                strict=strict,
                constraints=constraints,
                anchors=anchors,
                anchor_assignment=anchor_assignment,
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
            geometry_id=self.computational_geometry.geometry_id,
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
            geometry_id=self.computational_geometry.geometry_id,
            mesh_id=mesh_id(self.mesh),
            meshing_seconds=self._mesh_seconds,
            source=asdict(self.source),
        )
        result = Result.from_run(raw, self.computational_geometry, self.configuration())
        if not result.converged and require_converged:
            raise ConvergenceError(result)
        self._result = result
        return result

    def configuration(self):
        config = dict(
            size=self.size,
            fmin=self.fmin,
            fmax=self.fmax,
            settings=asdict(self.settings),
            angles=self._angles,
            pml=asdict(self.pml),
            layout=asdict(self.layout),
        )
        if self._automatic_domain:
            config["automatic_domain"] = dict(
                policy=asdict(self._domain_policy),
                input_geometry=self.geometry.as_dict(),
                coordinate_offset=self.coordinate_offset,
            )
        return config

    def plot_geometry(self, *, mesh=False, units="m", ax=None):
        from .plotting import geometry_plot

        if mesh and self.mesh is None:
            raise RuntimeError("Call apply_mesh before plotting grid lines")
        return geometry_plot(
            self.computational_geometry,
            self.mesh if mesh else None,
            self.configuration(),
            units,
            ax,
        )

    def plot_mesh(self, *, units="m", ax=None):
        from .plotting import mesh_plot

        return mesh_plot(self.mesh, self.wavelength, units, ax)

    def plot_discretization(self, *, units="m", ax=None):
        from .plotting import discretization_plot

        return discretization_plot(
            self.computational_geometry,
            self.mesh,
            self.discretization,
            self.configuration(),
            units,
            ax,
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
