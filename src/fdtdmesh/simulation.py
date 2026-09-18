"""Familiar scene-first FDTD API; physical coordinates are always metres."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .mesh import Mesh, cell_count, density_mesh
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

    def add_PML(self, *args, **kwargs):
        raise NotImplementedError(
            "Stage 1 supports PEC outer boundaries. CFS-CPML is a later stage."
        )

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
    ):
        probe = self.make_probe(kind, x, y)
        wave = Waveform(waveform, amplitude, frequency, delay, width, phase)
        wave.sample(np.array([0.0]), self.f_max)  # Validate at definition time.
        self.sources.append((probe, wave))
        return len(self.sources) - 1

    def add_receiver(self, kind, *, x, y):
        self.receivers.append(self.make_probe(kind, x, y))
        return len(self.receivers) - 1

    def add_line_monitor(self, *, x, y):
        """Stage-1 monitor records Ez at every node along the line."""
        return self.add_receiver("line", x=x, y=y)

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
        self.mesh = Mesh(*axes)
        self.Nx, self.Ny = nx, ny
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
            **constraints,
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
            self.mesh_uniform()
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
        for probe, wave in self.sources:
            ij = probe_indices(probe, self.mesh)
            values = wave.sample(times, self.f_max).astype(self.dtype)
            for i, j in ij:
                if c.pec[i, j]:
                    raise ValueError("Source maps to a PEC node or outer boundary")
                flat = int(i * (self.Ny + 1) + j)
                if flat not in source_map:
                    source_map[flat] = np.zeros(nt, dtype=self.dtype)
                source_map[flat] += values
        source_indices = np.array(sorted(source_map), dtype=np.int32)
        waves = (
            np.ascontiguousarray(np.stack([source_map[i] for i in source_indices], axis=1))
            if len(source_indices)
            else np.empty((nt, 0), dtype=self.dtype)
        )
        sites, slices, coordinates = [], {}, {}
        for receiver, probe in enumerate(self.receivers):
            ij = probe_indices(probe, self.mesh)
            if not len(ij):
                raise ValueError("Receiver does not intersect any mesh nodes")
            start = len(sites)
            sites.extend((ij[:, 0] * (self.Ny + 1) + ij[:, 1]).tolist())
            slices[receiver] = slice(start, len(sites))
            coordinates[receiver] = np.column_stack((self.mesh.x[ij[:, 0]], self.mesh.y[ij[:, 1]]))
        return (
            c,
            fields,
            source_indices,
            waves,
            np.array(sites, dtype=np.int32),
            slices,
            coordinates,
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
        )
        result = SimulationResult(
            self.mesh,
            c.dt,
            (np.arange(nt) + 1) * c.dt,
            {i: history[:, s].copy() for i, s in slices.items()},
            coordinates,
            {"Ez": ez, "Hx": hx, "Hy": hy},
            stats,
        )
        self.result = result
        self.Ez, self.Hx, self.Hy = ez, hx, hy
        return result
