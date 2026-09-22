"""Fixed vacuum collars and staggered CFS-CPML coefficient profiles."""

from dataclasses import dataclass

import numpy as np

from .constants import EPS0, MU0
from .mesh import AxisCollar, MeshInfeasibleError, cell_count


@dataclass(frozen=True)
class PML:
    x: AxisCollar
    y: AxisCollar
    order: int = 3
    kappa_max: float = 3.0
    alpha_max: float = 0.05  # Electric-equivalent conductivity, S/m.
    R0: float = 1e-8

    def __post_init__(self):
        object.__setattr__(self, "order", cell_count(self.order))
        if (
            not np.isfinite([self.kappa_max, self.alpha_max, self.R0]).all()
            or self.kappa_max < 1
            or self.alpha_max < 0
            or not 0 < self.R0 < 1
        ):
            raise ValueError("PML requires kappa_max>=1, alpha_max>=0, and 0<R0<1")
        if not self.x.cells and not self.y.cells:
            raise ValueError("Enable PML on at least one axis")

    def interfaces(self, scene):
        return (
            self.x.thickness,
            scene.Lx - self.x.thickness,
            self.y.thickness,
            scene.Ly - self.y.thickness,
        )

    def validate_scene(self, scene):
        """Continuous bounding checks prevent hidden/subpixel objects in vacuum PML."""
        x0, x1, y0, y1 = self.interfaces(scene)

        def inside(bounds):
            a, b, c, d = bounds
            return (not self.x.cells or (a > x0 and b < x1)) and (
                not self.y.cells or (c > y0 and d < y1)
            )

        for kind, _, data in scene.primitives:
            if kind == "circle":
                x, y, r = data
                bounds = x - r, x + r, y - r, y + r
            elif kind == "polygon":
                bounds = data[:, 0].min(), data[:, 0].max(), data[:, 1].min(), data[:, 1].max()
            else:
                bounds = probe_bounds(data)
            if not inside(bounds):
                raise ValueError("Geometry must lie strictly inside the vacuum PML interfaces")
        for probe in [s[0] for s in scene.sources] + scene.receivers:
            if not inside(probe_bounds(probe)):
                raise ValueError("Sources and receivers must lie strictly inside PML interfaces")

    def validate_mesh(self, mesh):
        for axis, collar in (("x", self.x), ("y", self.y)):
            lines = getattr(mesh, axis)
            for index, value in collar.fixed_lines(lines[-1], len(lines) - 1).items():
                if lines[index] != value:
                    raise MeshInfeasibleError(
                        f"{axis} mesh changes a fixed PML line at index {index}"
                    )


def probe_bounds(probe):
    x, y = np.atleast_1d(probe.x), np.atleast_1d(probe.y)
    return x.min(), x.max(), y.min(), y.max()


def profile(coordinates, length, collar, pml, dt):
    """Rows are [1/kappa, b, a]; psi_new = b*psi + a*field_difference.

    Electric and magnetic profiles use the same electric-equivalent damping rates,
    sampled at their actual Yee coordinates (impedance-matched vacuum collars).
    """
    out = np.zeros((len(coordinates), 3))
    out[:, :2] = 1
    if not collar.cells:
        return out
    t = collar.thickness
    u = np.maximum((t - coordinates) / t, (coordinates - (length - t)) / t).clip(0, 1)
    active = u > 0
    sigma_max = -(pml.order + 1) * np.log(pml.R0) / (2 * np.sqrt(MU0 / EPS0) * t)
    sigma = sigma_max * u**pml.order
    kappa = 1 + (pml.kappa_max - 1) * u**pml.order
    alpha = pml.alpha_max * (1 - u) * active
    exponent = -(sigma / kappa + alpha) * dt / EPS0
    b = np.exp(exponent)
    a = np.zeros_like(u)
    denominator = sigma * kappa + alpha * kappa * kappa
    np.divide(sigma * np.expm1(exponent), denominator, out=a, where=denominator > 0)
    out[:, 0], out[:, 1], out[:, 2] = 1 / kappa, b, a
    return out


def build_cpml(scene, mesh, dt, dtype):
    pml = getattr(scene, "pml", None)
    if pml is None:
        return np.empty((0, 3), dtype=dtype)
    pml.validate_scene(scene)
    pml.validate_mesh(mesh)
    return np.ascontiguousarray(
        np.concatenate(
            [
                profile(mesh.x, scene.Lx, pml.x, pml, dt),
                profile(mesh.y, scene.Ly, pml.y, pml, dt),
                profile((mesh.x[:-1] + mesh.x[1:]) / 2, scene.Lx, pml.x, pml, dt),
                profile((mesh.y[:-1] + mesh.y[1:]) / 2, scene.Ly, pml.y, pml, dt),
            ]
        ),
        dtype=dtype,
    )
