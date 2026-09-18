"""Familiar scene-first FDTD API; physical coordinates are always metres."""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .mesh import AxisCollar, Mesh, cell_count, density_mesh
from .pml import PML
from .sampling import dual_areas, point_stencil, receiver_points
from .scene import Scene2D, probe_indices
from .solver.coefficients import build_coefficients
from .sources import Waveform


@dataclass
class SimulationResult:
    mesh: Mesh
    dt: float
    times: np.ndarray
    receivers: dict
    receiver_coordinates: dict
    fields: dict
    diagnostics: dict

    def resample(self, receiver, times):
        """Interpolate onto common physical times, refusing extrapolation."""
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or np.any(times < self.times[0])
            or np.any(times > self.times[-1])
        ):
            raise ValueError("Requested times must be increasing and within recorded time range")
        return np.column_stack(
            [np.interp(times, self.times, column) for column in self.receivers[receiver].T]
        )

    def spectrum(self, receiver, frequencies):
        """Unwindowed time-integral DFT of Ez after the run, shape (frequency, probe)."""
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if (
            frequencies.ndim != 1
            or not np.isfinite(frequencies).all()
            or np.any(frequencies < 0)
            or np.any(frequencies > 0.5 / self.dt)
        ):
            raise ValueError("DFT frequencies must lie between zero and Nyquist")
        return (
            self.dt
            * np.exp(-2j * np.pi * frequencies[:, None] * self.times)
            @ self.receivers[receiver]
        )


class FDTD_2D_Ez(Scene2D):
    def __init__(
        self,
        x_range,
        y_range,
        Nx=None,
        Ny=None,
        f_max=None,
        Nt=None,
        f_min=None,
        dt=None,
        t_end=None,
        dtype="float32",
    ):
        super().__init__(x_range, y_range)
        self.Nx = None if Nx is None else cell_count(Nx)
        self.Ny = None if Ny is None else cell_count(Ny)
        if f_max is None or not np.isfinite(f_max) or f_max <= 0:
            raise ValueError("f_max must be finite and positive")
        self.f_max = float(f_max)
        self.f_min = 0.0 if f_min is None else float(f_min)
        if not np.isfinite(self.f_min) or not 0 <= self.f_min <= self.f_max:
            raise ValueError("Require 0 <= f_min <= f_max")
        self.Nt = None if Nt is None else cell_count(Nt)
        if t_end is not None and (not np.isfinite(t_end) or t_end <= 0):
            raise ValueError("t_end must be finite and positive")
        if Nt is not None and t_end is not None:
            raise ValueError("Specify Nt or t_end, not both")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        self.t_end, self.dt, self.dtype = t_end, dt, dtype
        self.mesh = None
        self._mesh_model = None

    def config(self, backend="gpu"):
        if backend not in ("gpu", "cuda"):
            raise ValueError("Production backend is CUDA only; NumPy oracle is test-only")
        return self

    def add_PML(
        self,
        pml_width=10,
        *,
        thickness=None,
        direction="xy",
        order=3,
        kappa_max=3.0,
        alpha_max=0.05,
        R0=1e-8,
    ):
        """Reserve symmetric vacuum collars inside Nx/Ny; thickness is in metres.

        If omitted, thickness is width * initial uniform spacing and is frozen
        here, so subsequent CNN budgets cannot change the physical PML domain.
        """
        if self.pml is not None:
            raise ValueError("PML is already configured; create a new simulation to change collars")
        if direction not in ("x", "y", "xy"):
            raise ValueError("direction must be x, y, or xy")
        widths = [pml_width, pml_width] if np.ndim(pml_width) == 0 else list(pml_width)
        if len(widths) != 2:
            raise ValueError("pml_width must be a count or (x_count,y_count)")
        if thickness is None:
            nx, ny = self._budget()
            thicknesses = [widths[0] * self.Lx / nx, widths[1] * self.Ly / ny]
        else:
            thicknesses = [thickness, thickness] if np.ndim(thickness) == 0 else list(thickness)
        if len(thicknesses) != 2:
            raise ValueError("thickness must be metres or (x_metres,y_metres)")
        collars = [
            AxisCollar(cell_count(w), float(t)) if axis in direction else AxisCollar()
            for axis, w, t in zip(("x", "y"), widths, thicknesses)
        ]
        for collar, length, count in zip(collars, (self.Lx, self.Ly), (self.Nx, self.Ny)):
            if count is not None:
                collar.fixed_lines(length, count)
            elif 2 * collar.thickness >= length:
                raise ValueError("PML thickness leaves no interior")
        pml = PML(*collars, order=order, kappa_max=kappa_max, alpha_max=alpha_max, R0=R0)
        pml.validate_scene(self)
        for axis, collar, length in zip(("x", "y"), collars, (self.Lx, self.Ly)):
            if collar.cells:
                self.add_anchor(axis, collar.thickness)
                self.add_anchor(axis, length - collar.thickness)
        self.pml = pml
        self.mesh = None
        return self

    def _mesh_options(self, options):
        options = dict(options)
        if self.pml is not None:
            self.pml.validate_scene(self)
            if "x_collar" in options or "y_collar" in options:
                raise ValueError("PML collars are controlled by add_PML")
            options.update(x_collar=self.pml.x, y_collar=self.pml.y)
        return options

    def add_source(
        self,
        kind,
        *,
        x,
        y,
        waveform="gaussian",
        amplitude=1.0,
        frequency=None,
        delay=None,
        width=None,
        phase=0.0,
        normalization="field_increment",
    ):
        probe = self.make_probe(kind, x, y)
        if normalization == "current" and probe.kind != "point":
            raise ValueError(
                "Integrated current sources are point sources; use current_density for lines"
            )
        wave = Waveform(waveform, amplitude, frequency, delay, width, phase, normalization)
        wave.sample(np.array([0.0]), self.f_max)  # Validate at definition time.
        self.sources.append((probe, wave))
        return len(self.sources) - 1

    def add_receiver(self, kind, *, x, y, samples=None):
        probe = self.make_probe(kind, x, y)
        if samples is not None:
            samples = cell_count(samples)
            if probe.kind != "line" or samples < 2:
                raise ValueError("samples requires a line receiver and at least two points")
            probe = replace(probe, samples=samples)
        self.receivers.append(probe)
        return len(self.receivers) - 1

    def add_line_monitor(self, *, x, y, samples=None):
        """Record Ez at fixed physical samples or every mesh node along the line."""
        return self.add_receiver("line", x=x, y=y, samples=samples)

    def _budget(self, Nx=None, Ny=None):
        nx = self.Nx if Nx is None else cell_count(Nx)
        ny = self.Ny if Ny is None else cell_count(Ny)
        if nx is None or ny is None:
            raise ValueError("Specify Nx and Ny before generating a mesh")
        return nx, ny

    def _validate_mesh(self, mesh):
        if mesh.x[-1] != self.Lx or mesh.y[-1] != self.Ly:
            raise ValueError("Mesh boundaries must match the physical domain exactly")
        for axis in ("x", "y"):
            lines = getattr(mesh, axis)
            missing = [a for a in getattr(self, f"{axis}_anchors") if not np.any(lines == a)]
            if missing:
                raise ValueError(
                    f"Mesh is missing mandatory {axis} anchors {missing}; remesh the scene"
                )
        if self.pml is not None:
            self.pml.validate_scene(self)
            self.pml.validate_mesh(mesh)

    def set_mesh(self, x_lines, y_lines):
        mesh = Mesh(x_lines, y_lines)
        if (self.Nx is not None and mesh.Nx != self.Nx) or (
            self.Ny is not None and mesh.Ny != self.Ny
        ):
            raise ValueError("Supplied mesh must match the exact Nx/Ny budget")
        self._validate_mesh(mesh)
        self.Nx, self.Ny, self.mesh = mesh.Nx, mesh.Ny, mesh
        return mesh

    def mesh_uniform(self, *, Nx=None, Ny=None):
        nx, ny = self._budget(Nx, Ny)
        mesh = Mesh(np.linspace(0, self.Lx, nx + 1), np.linspace(0, self.Ly, ny + 1))
        # Accept numerically identical uniform anchors, replace with exact supplied value.
        axes = []
        for axis, length in (("x", self.Lx), ("y", self.Ly)):
            lines = getattr(mesh, axis).copy()
            if self.pml is not None:
                collar = getattr(self.pml, axis)
                for i, value in collar.fixed_lines(length, len(lines) - 1).items():
                    if abs(lines[i] - value) > 1e-12 * length:
                        raise ValueError(
                            "Uniform mesh cannot retain fixed PML collars; use mesh_from_density"
                        )
                    lines[i] = value
            assigned = set()
            for anchor in sorted(getattr(self, f"{axis}_anchors")):
                i = int(np.argmin(abs(lines - anchor)))
                if i in assigned or abs(lines[i] - anchor) > 1e-12 * length:
                    raise ValueError(
                        "Uniform mesh cannot retain these anchors; use mesh_from_density"
                    )
                lines[i] = anchor
                assigned.add(i)
            axes.append(lines)
        mesh = Mesh(*axes)
        self._validate_mesh(mesh)
        self.Nx, self.Ny, self.mesh = nx, ny, mesh
        return self.mesh

    def mesh_from_density(self, rho_x, rho_y, *, Nx=None, Ny=None, **constraints):
        nx, ny = self._budget(Nx, Ny)
        mesh = density_mesh(
            self.Lx,
            self.Ly,
            nx,
            ny,
            rho_x,
            rho_y,
            x_anchors=sorted(self.x_anchors),
            y_anchors=sorted(self.y_anchors),
            **self._mesh_options(constraints),
        )
        self.Nx, self.Ny, self.mesh = nx, ny, mesh
        return mesh

    def load_mesh_model(self, path, *, device="cpu"):
        from .ml import load_model

        self._mesh_model = load_model(path, device=device)
        return self

    def mesh_with_model(self, path=None, *, Nx=None, Ny=None, device="cpu", **constraints):
        from .ml import infer_mesh

        if path is not None:
            self.load_mesh_model(path, device=device)
        if self._mesh_model is None:
            raise ValueError("Load a model or supply a checkpoint path")
        nx, ny = self._budget(Nx, Ny)
        mesh = infer_mesh(self, *self._mesh_model, nx, ny, self.f_max, self.f_min, **constraints)
        self.Nx, self.Ny, self.mesh = nx, ny, mesh
        return mesh

    def _prepare(self, initial_fields=None):
        if self.mesh is None:
            if self.pml is None:
                self.mesh_uniform()
            else:
                self.mesh_from_density([1], [1])
        self._validate_mesh(self.mesh)
        c = build_coefficients(self, self.mesh, dt=self.dt, dtype=self.dtype)
        if self.t_end is None and self.Nt is None:
            raise ValueError("Specify a fixed physical t_end or Nt")
        nt = int(np.ceil(self.t_end / c.dt)) if self.t_end is not None else self.Nt
        if nt > np.iinfo(np.int32).max:
            raise ValueError("Number of timesteps exceeds native runtime capacity")
        times = (np.arange(nt) + 1) * c.dt
        initial_fields = initial_fields or {}
        if set(initial_fields) - {"Ez", "Hx", "Hy"}:
            raise ValueError("Initial fields may only contain Ez, Hx, Hy")
        fields = []
        for name, shape in (
            ("Ez", (self.Nx + 1, self.Ny + 1)),
            ("Hx", (self.Nx + 1, self.Ny)),
            ("Hy", (self.Nx, self.Ny + 1)),
        ):
            a = np.array(initial_fields.get(name, np.zeros(shape)), dtype=self.dtype, order="C")
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"Initial {name} must be finite with shape {shape}")
            fields.append(a)
        fields[0][c.pec.astype(bool)] = 0
        # Sum all overlapping source waveforms per unique Ez node before upload.
        source_map = {}
        areas = dual_areas(self.mesh).ravel()
        for probe, wave in self.sources:
            if wave.normalization == "current":
                sites, weights = point_stencil(self.mesh, probe.x, probe.y)
                weights = weights / areas[sites]
            else:
                ij = probe_indices(probe, self.mesh)
                sites, weights = ij[:, 0] * (self.Ny + 1) + ij[:, 1], np.ones(len(ij))
            sample_times = times if wave.normalization == "field_increment" else times - c.dt / 2
            values = wave.sample(sample_times, self.f_max).astype(self.dtype)
            for flat, weight in zip(sites, weights):
                if weight == 0:
                    continue
                i, j = divmod(int(flat), self.Ny + 1)
                if c.pec[i, j]:
                    raise ValueError("Source maps to a PEC node or outer boundary")
                self._check_pml_support(i, j)
                scale = (
                    weight
                    if wave.normalization == "field_increment"
                    else -weight * c.current_scale[i, j]
                )
                if flat not in source_map:
                    source_map[flat] = np.zeros(nt, dtype=self.dtype)
                source_map[flat] += values * scale
        source_indices = np.array(sorted(source_map), dtype=np.int32)
        waves = (
            np.ascontiguousarray(np.stack([source_map[i] for i in source_indices], axis=1))
            if len(source_indices)
            else np.empty((nt, 0), dtype=self.dtype)
        )
        sites, slices, coordinates = [], {}, {}
        for receiver, probe in enumerate(self.receivers):
            points = receiver_points(probe, self.mesh)
            if not len(points):
                raise ValueError("Receiver does not intersect any mesh nodes")
            start = len(sites)
            weights = []
            for x, y in points:
                indices, w = point_stencil(self.mesh, x, y)
                for site, weight in zip(indices, w):
                    if weight:
                        self._check_pml_support(*divmod(int(site), self.Ny + 1))
                sites.extend(indices)
                weights.append(w)
            slices[receiver] = (slice(start, len(sites)), np.array(weights, dtype=self.dtype))
            coordinates[receiver] = points
        return (
            c,
            fields,
            source_indices,
            waves,
            np.array(sites, dtype=np.int32),
            slices,
            coordinates,
        )

    def _check_pml_support(self, i, j):
        if self.pml is None:
            return
        x0, x1, y0, y1 = self.pml.interfaces(self)
        if not x0 <= self.mesh.x[i] <= x1 or not y0 <= self.mesh.y[j] <= y1:
            raise ValueError(
                "Source/receiver interpolation support enters PML; move it farther inside"
            )

    def run(self, *, initial_fields=None):
        from .solver.tmz import cuda_backend

        runtime = cuda_backend()
        start = perf_counter()
        c, fields, sources, waves, receivers, slices, coordinates = self._prepare(initial_fields)
        ez, hx, hy, history, stats = runtime.run(c, fields, sources, waves, receivers)
        nt = len(waves)
        stats.update(
            self.mesh.diagnostics(),
            dt_cfl=c.dt_cfl,
            dt=c.dt,
            Nt=nt,
            simulated_time=nt * c.dt,
            requested_time=self.t_end,
            cell_updates=self.Nx * self.Ny * nt,
            wall_seconds=perf_counter() - start,
            backend="cuda",
            boundary="CFS-CPML" if self.pml else "PEC",
            source_normalizations=[wave.normalization for _, wave in self.sources],
        )
        result = SimulationResult(
            self.mesh,
            c.dt,
            (np.arange(nt) + 1) * c.dt,
            {
                i: (history[:, s].reshape(nt, -1, 4) * weights[None]).sum(axis=2)
                for i, (s, weights) in slices.items()
            },
            coordinates,
            {"Ez": ez, "Hx": hx, "Hy": hy},
            stats,
        )
        self.result = result
        self.Ez, self.Hx, self.Hy = ez, hx, hy
        return result
