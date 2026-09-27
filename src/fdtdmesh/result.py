"""Plot-ready immutable result snapshots, usable without a CUDA runtime."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def readonly(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def json_text(value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    return json.dumps(clean(value), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ConvergenceHistory:
    steps: np.ndarray
    times: np.ndarray
    current_error_ratio: np.ndarray
    incident_error_ratio: np.ndarray
    incident_strength: np.ndarray
    residual: np.ndarray
    stable_checks: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class Result:
    frequencies: np.ndarray
    angles: np.ndarray
    far_field: np.ndarray
    scattering_width: np.ndarray
    history: np.ndarray
    bin_history: np.ndarray
    geometry: object
    mesh: object
    _diagnostics_json: str
    _configuration_json: str
    _debug: object = None

    def __post_init__(self):
        for name in (
            "frequencies",
            "angles",
            "far_field",
            "scattering_width",
            "history",
            "bin_history",
        ):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        if (
            self.far_field.shape != (len(self.frequencies), len(self.angles))
            or self.scattering_width.shape != self.far_field.shape
        ):
            raise ValueError("Invalid far-field result shapes")
        if (
            self.history.ndim != 2
            or self.history.shape[1] != 8
            or self.bin_history.shape != (len(self.history), len(self.frequencies), 3)
        ):
            raise ValueError("Invalid convergence history shapes")

    @classmethod
    def from_run(cls, raw, geometry, configuration):
        from types import MappingProxyType

        debug = MappingProxyType({k: readonly(v) for k, v in raw.debug.items()})
        return cls(
            raw.frequencies,
            raw.angles,
            raw.amplitude,
            raw.width,
            raw.history,
            raw.bin_history,
            geometry,
            raw.mesh,
            json_text(raw.diagnostics),
            json_text(configuration),
            debug,
        )

    @property
    def amplitude(self):
        return self.far_field

    @property
    def width(self):
        return self.scattering_width

    @property
    def diagnostics(self):
        return json.loads(self._diagnostics_json)

    @property
    def configuration(self):
        return json.loads(self._configuration_json)

    @property
    def input_geometry(self):
        """Exact original-coordinate recipe, including before automatic translation."""
        from .geometry import Geometry

        automatic = self.configuration.get("automatic_domain")
        return Geometry.from_dict(automatic["input_geometry"]) if automatic else self.geometry

    @property
    def debug(self):
        return {} if self._debug is None else self._debug

    @property
    def converged(self):
        return self.diagnostics["status"] == "converged"

    @property
    def convergence(self):
        h, b = self.history, self.bin_history
        return ConvergenceHistory(
            h[:, 0], h[:, 1], b[:, :, 0], b[:, :, 1], b[:, :, 2], h[:, 3], h[:, 4], h[:, 5]
        )

    def save(self, path):
        from .storage import save_result

        save_result(self, path)

    @classmethod
    def load(cls, path):
        from .storage import load_result

        return load_result(path)

    def plot_geometry(self, *, mesh=False, units="m", ax=None):
        from .plotting import geometry_plot

        return geometry_plot(
            self.geometry, self.mesh if mesh else None, self.configuration, units, ax
        )

    def plot_mesh(self, *, units="m", ax=None):
        from .constants import C0
        from .plotting import mesh_plot

        c = self.configuration
        return mesh_plot(self.mesh, C0 / ((c["fmin"] + c["fmax"]) / 2), units, ax)

    def plot_convergence(self, *, per_bin=True, ax=None):
        from .plotting import convergence_plot

        return convergence_plot(self, per_bin, ax)

    def plot_scattering(self, *, frequency=None, scale="linear", normalize="wavelength", ax=None):
        from .plotting import scattering_plot

        return scattering_plot(self, frequency, scale, normalize, ax)

    def plot_far_field(self, *, frequency=None, component="magnitude", ax=None):
        from .plotting import far_field_plot

        return far_field_plot(self, frequency, component, ax)

    def save_plots(self, directory):
        """Write six PNGs with a headless canvas, preserving any interactive backend."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        f = self.frequencies[len(self.frequencies) // 2]
        factories = {
            "geometry": lambda ax: self.plot_geometry(mesh=True, units="wavelength", ax=ax),
            "mesh": lambda ax: self.plot_mesh(units="wavelength", ax=ax),
            "convergence": lambda ax: self.plot_convergence(ax=ax),
            "scattering": lambda ax: self.plot_scattering(frequency=f, ax=ax),
            "amplitude": lambda ax: self.plot_far_field(frequency=f, ax=ax),
            "phase": lambda ax: self.plot_far_field(frequency=f, component="phase", ax=ax),
        }
        paths = []
        for name, make in factories.items():
            fig = Figure(figsize=(7, 5))
            FigureCanvasAgg(fig)
            angular = name in ("scattering", "amplitude", "phase")
            make(fig.subplots(subplot_kw={"projection": "polar"} if angular else {}))
            target = path / f"{name}.png"
            fig.savefig(target, dpi=160, bbox_inches="tight")
            fig.clear()
            paths.append(target)
        return paths
