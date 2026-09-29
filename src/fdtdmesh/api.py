"""Define simulation -> add exact geometry -> apply mesh -> solve -> save/plot."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .boundary import BoundaryPolicy
from .constants import C0
from .domain import DomainPolicy
from .geometry import Geometry
from .mesh import Mesh
from .pulse import GaussianPulse
from .simulation import Convergence, ConvergenceError, ScatteringCase, prepare, run_scattering
from .strategies import generate_mesh, mesh_id

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
    margin_mesh: str = "fixed"

    def __post_init__(self):
        if self.margin_mesh not in ("fixed", "graded"):
            raise ValueError("margin_mesh must be fixed or graded")
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
        self, *, fmin, fmax, solver=None, observation_angles_deg=None, domain=None, boundary=None
    ):
        angles = None if observation_angles_deg is None else np.deg2rad(observation_angles_deg)
        self._initialize(fmin=fmin, fmax=fmax, settings=solver, angles=angles, domain=domain)
        self._boundary = boundary or BoundaryPolicy()
        if not isinstance(self._boundary, BoundaryPolicy):
            raise TypeError("boundary must be BoundaryPolicy")

    @classmethod
    def _from_resolved(
        cls, size, fmin, fmax, *, pml, layout, settings=None, angles=None, boundary=None
    ):
        """Internal explicit layout for refinement, sensitivity tests and archive reads."""
        obj = cls(fmin=fmin, fmax=fmax, solver=settings, boundary=boundary)
        obj._automatic_domain = False
        obj._domain_policy = None
        obj._geometry = Geometry(tuple(size))
        obj._pml, obj._layout = pml, layout
        if angles is not None:
            obj._angles = tuple(np.asarray(angles, float).tolist())
        return obj

    def _initialize(self, *, fmin, fmax, settings=None, angles=None, domain=None):
        if domain is not None and not isinstance(domain, DomainPolicy):
            raise TypeError("domain must be DomainPolicy")
        self._automatic_domain = True
        self._domain_policy = domain or DomainPolicy()
        self._resolved_geometry = None
        self._coordinate_offset = (0.0, 0.0)
        self._geometry = Geometry()
        self._pulse = GaussianPulse(float(fmin), float(fmax))
        self._settings = settings or SolverSettings()
        if not isinstance(self._settings, SolverSettings):
            raise TypeError("solver must be SolverSettings")
        self._settings.frequencies(fmin, fmax)
        angles = np.deg2rad(np.arange(360)) if angles is None else np.asarray(angles, float)
        if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
            raise ValueError("Observation angles must be a finite nonempty vector")
        self._angles = tuple(angles.tolist())
        self._mesh = self._prepared = self._case = self._result = None
        self._mesh_seconds = 0.0
        self._pml = self._layout = None

    @property
    def geometry(self):
        return self._geometry

    @property
    def size(self):
        return self.computational_geometry.size

    def _resolve_domain(self):
        if self._automatic_domain and self._resolved_geometry is None:
            values = self._domain_policy.resolve(self.geometry, self.fmin, self.fmax)
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
            boundary=self._boundary,
        )

    def apply_mesh(
        self,
        strategy="geometry_aware",
        *,
        cells=None,
        target_spacing=None,
        max_cells=(512, 512),
        mesh=None,
        density=None,
        options=None,
    ):
        """Prepare axes and boundary operators. Fixed cells are an exact total budget."""
        from .meshing import MeshOptions

        opts = options or MeshOptions()
        if not isinstance(opts, MeshOptions):
            raise TypeError("options must be MeshOptions")
        if strategy not in ("uniform", "quasi_uniform", "geometry_aware", "density", "custom"):
            raise ValueError(
                "Unknown grid strategy; use uniform, quasi_uniform, geometry_aware, density or custom"
            )
        if strategy == "custom":
            if (
                mesh is None
                or any(v is not None for v in (cells, target_spacing, density, options))
                or max_cells != (512, 512)
            ):
                raise ValueError("custom accepts only mesh=")
            return self._apply_mesh(mesh)
        if mesh is not None:
            raise ValueError("mesh= requires custom strategy")
        if strategy == "geometry_aware":
            if density is not None or opts.anchors is not None or opts.anchor_assignment != "local":
                raise ValueError(
                    "geometry_aware selects its own feature anchors; density/anchor overrides require density strategy"
                )
            if cells is not None and (target_spacing is not None or max_cells != (512, 512)):
                raise ValueError("Choose exact cells OR target_spacing/max_cells")
            return self._apply_mesh(
                strategy,
                cells=cells,
                target_spacing=target_spacing,
                max_cells=max_cells,
                constraints=opts.constraints,
                time_limit=opts.time_limit,
                max_passes=opts.max_passes,
                hybrid_repair_passes=opts.hybrid_repair_passes,
            )
        if target_spacing is not None or max_cells != (512, 512):
            raise ValueError("Automatic counts require geometry_aware")
        if density is not None and strategy != "density":
            raise ValueError("density= requires density strategy")
        candidate = self._apply_mesh(
            "uniform" if strategy == "quasi_uniform" else strategy,
            cells=cells,
            density=density,
            strict=strategy == "uniform",
            constraints=opts.constraints,
            anchors=opts.anchors,
            anchor_assignment=opts.anchor_assignment,
            time_limit=opts.time_limit,
        )
        candidate.metadata["strategy"] = strategy
        return candidate

    def _apply_mesh(
        self,
        strategy="uniform",
        *,
        cells=None,
        density=None,
        strict=False,
        constraints=None,
        anchors=None,
        anchor_assignment="joint",
        time_limit=30.0,
        target_spacing=None,
        max_cells=(512, 512),
        max_passes=20,
        hybrid_repair_passes=3,
    ):
        """Prepare a grid and conformal enlarged-cell coefficients; return the Mesh.

        Internal strategy: uniform target, deterministic research baseline, density, geometry_aware, or Mesh.
        cells includes PML. strict=True requires an exactly uniform grid.
        A failed proposal leaves the previously applied mesh intact.
        geometry_aware selects counts using target_spacing, max_cells, max_passes,
        and the construction time_limit. See docs/mesh_strategy.md.
        """
        start = perf_counter()
        metadata = {}
        if isinstance(strategy, Mesh):
            if (
                cells is not None
                or density is not None
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

            if density is not None or strict or anchors is not None or anchor_assignment != "joint":
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
                cells=cells,
                boundary=self._boundary,
                hybrid_repair_passes=hybrid_repair_passes,
            )
        else:
            if target_spacing is not None or max_cells != (512, 512) or max_passes != 20:
                raise ValueError("target_spacing/max_cells/max_passes require geometry_aware")
            if cells is None:
                raise ValueError("Specify cells=(Nx, Ny), including fixed PML collars")
            actual_strategy = strategy
            candidate = generate_mesh(
                self.computational_geometry,
                self.layout,
                self.pml,
                cells,
                actual_strategy,
                density=density,
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
        candidate.metadata["preparation"] = dict(
            geometry_id=self.computational_geometry.geometry_id,
            boundary=asdict(self._boundary),
            witness_anchors=candidate.metadata.get("geometry_aware", {}).get(
                "witness_anchors", [[], []]
            ),
        )
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
            boundary=asdict(self._boundary),
        )
        if self._automatic_domain:
            allocation = self._domain_policy.allocation(self.fmin, self.fmax)
            data["domain_allocation"] = allocation
            reserve = 2 * sum(allocation["cells"])
            data["reserved_axis_cells"] = [reserve, reserve]
            if self.mesh is not None:
                data["scatterer_axis_cells"] = [self.mesh.Nx - reserve, self.mesh.Ny - reserve]
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
            boundary=self._prepared[0].boundary_report,
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
            boundary=asdict(self._boundary),
            pml=asdict(self.pml),
            layout=asdict(self.layout),
        )
        if self._automatic_domain:
            config["domain_allocation"] = self._domain_policy.allocation(self.fmin, self.fmax)
            config["automatic_domain"] = dict(
                policy=asdict(self._domain_policy),
                input_geometry=self.geometry.as_dict(),
                coordinate_offset=self.coordinate_offset,
            )
        return config

    def plot_geometry(self, *, mesh=False, units="m", ax=None, figsize=(16, 14)):
        from .plotting import geometry_plot

        if mesh and self.mesh is None:
            raise RuntimeError("Call apply_mesh before plotting grid lines")
        fig = geometry_plot(
            self.computational_geometry,
            self.mesh if mesh else None,
            self.configuration(),
            units,
            ax,
        )
        if ax is None:
            fig.set_size_inches(*figsize)
        return fig

    def plot_mesh(self, *, units="m", ax=None):
        from .plotting import mesh_plot

        return mesh_plot(self.mesh, self.wavelength, units, ax)

    def plot_discretization(self, *, units="m", ax=None, show_fallback=True, figsize=(16, 14)):
        from .plotting import discretization_plot

        fig = discretization_plot(
            self.computational_geometry,
            self.mesh,
            self.discretization,
            self.configuration(),
            units,
            ax,
            show_fallback=show_fallback,
        )
        if ax is None:
            fig.set_size_inches(*figsize)
        return fig

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
