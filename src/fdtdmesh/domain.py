"""Geometry-first domain allocation, independent of interior mesh resolution."""

from dataclasses import dataclass

import numpy as np

from .constants import C0
from .geometry import Geometry
from .mesh import AxisCollar, cell_count
from .pml import PML


@dataclass(frozen=True)
class DomainPolicy:
    exterior_spacing: float | None = None
    pml_cells: int = 12
    pml_to_contour: int = 6
    contour_to_tfsf: int = 4
    scatterer_to_tfsf: int = 5
    phase_origin: tuple | None = None

    def __post_init__(self):
        for name, minimum in (
            ("pml_cells", 1),
            ("pml_to_contour", 2),
            ("contour_to_tfsf", 2),
            ("scatterer_to_tfsf", 3),
        ):
            n = cell_count(getattr(self, name))
            if n < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            object.__setattr__(self, name, n)
        if self.exterior_spacing is not None and (
            not np.isfinite(self.exterior_spacing) or self.exterior_spacing <= 0
        ):
            raise ValueError("exterior_spacing must be positive and finite")
        if self.phase_origin is not None:
            origin = tuple(float(v) for v in self.phase_origin)
            if len(origin) != 2 or not np.isfinite(origin).all():
                raise ValueError("phase_origin must be a finite coordinate pair")
            object.__setattr__(self, "phase_origin", origin)

    def resolve(self, geometry, fmax):
        from .api import ScatteringLayout

        bounds = geometry.bounds
        if bounds is None:
            raise ValueError("Automatic domain requires nonempty final PEC geometry")
        h = self.exterior_spacing or C0 / fmax / 24
        n = self.pml_cells + self.pml_to_contour + self.contour_to_tfsf + self.scatterer_to_tfsf
        low, high = np.array(bounds)[[0, 2]], np.array(bounds)[[1, 3]]
        offset = n * h - low
        size = tuple(high - low + 2 * n * h)
        if np.any(high <= low):
            raise ValueError("Final PEC geometry must have positive width and height")
        # Discard a design canvas, if supplied; only final material sets the domain.
        design = Geometry(None, geometry.shapes, geometry.next_id)
        resolved = design.translated(offset, size=size)
        collar = AxisCollar(self.pml_cells, self.pml_cells * h)
        c = (self.pml_cells + self.pml_to_contour) * h
        t = c + self.contour_to_tfsf * h
        origin = (
            (low + high) / 2 if self.phase_origin is None else np.asarray(self.phase_origin)
        ) + offset
        if not (t < origin[0] < size[0] - t and t < origin[1] < size[1] - t):
            raise ValueError("DomainPolicy.phase_origin must lie inside the derived TFSF box")
        layout = ScatteringLayout(
            (t, size[0] - t, t, size[1] - t),
            (c, size[0] - c, c, size[1] - c),
            (self.pml_cells + self.pml_to_contour // 2) * h,
            tuple(origin),
            exterior_cells=(self.pml_to_contour, self.contour_to_tfsf),
            scatterer_margin_cells=self.scatterer_to_tfsf,
        )
        return resolved, PML(collar, collar), layout, tuple(offset)
