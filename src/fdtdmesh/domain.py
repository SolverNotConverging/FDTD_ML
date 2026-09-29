"""Geometry-first domain allocation, independent of interior mesh resolution."""

from dataclasses import dataclass

import numpy as np

from .constants import C0
from .geometry import Geometry
from .mesh import AxisCollar, cell_count
from .pml import PML


@dataclass(frozen=True)
class DomainPolicy:
    scatterer_to_tfsf_wavelengths: float = 5 / 24
    tfsf_to_contour_wavelengths: float = 4 / 24
    contour_to_pml_wavelengths: float = 6 / 24
    pml_thickness_wavelengths: float = 0.5
    exterior_ppw: float = 24.0
    exterior_max_spacing: float | None = None
    min_pml_cells: int = 12
    phase_origin: tuple | None = None
    margin_mesh: str = "fixed"

    def __post_init__(self):
        if self.margin_mesh not in ("fixed", "graded"):
            raise ValueError("margin_mesh must be fixed or graded")
        for name in (
            "scatterer_to_tfsf_wavelengths",
            "tfsf_to_contour_wavelengths",
            "contour_to_pml_wavelengths",
            "pml_thickness_wavelengths",
            "exterior_ppw",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "min_pml_cells", cell_count(self.min_pml_cells))
        if self.exterior_max_spacing is not None and (
            not np.isfinite(self.exterior_max_spacing) or self.exterior_max_spacing <= 0
        ):
            raise ValueError("exterior_max_spacing must be finite and positive")
        if self.phase_origin is not None:
            origin = tuple(float(v) for v in self.phase_origin)
            if len(origin) != 2 or not np.isfinite(origin).all():
                raise ValueError("phase_origin must be a finite coordinate pair")
            object.__setattr__(self, "phase_origin", origin)

    def allocation(self, fmin, fmax):
        wavelength = C0 / ((fmin + fmax) / 2)
        requested = wavelength * np.array(
            [
                self.pml_thickness_wavelengths,
                self.contour_to_pml_wavelengths,
                self.tfsf_to_contour_wavelengths,
                self.scatterer_to_tfsf_wavelengths,
            ]
        )
        minimum = np.array([self.min_pml_cells, 2, 2, 3])
        h = min(
            self.exterior_max_spacing or C0 / fmax / self.exterior_ppw,
            float(np.min(requested / minimum)),
        )
        counts = np.maximum(minimum, np.ceil(requested / h - 1e-12).astype(int))
        return dict(
            wavelength=wavelength,
            spacing=h,
            requested=requested.tolist(),
            achieved=(counts * h).tolist(),
            cells=counts.tolist(),
            order=["pml", "contour_to_pml", "tfsf_to_contour", "scatterer_to_tfsf"],
        )

    def resolve(self, geometry, fmin, fmax):
        from .api import ScatteringLayout

        bounds = geometry.bounds
        if bounds is None:
            raise ValueError("Automatic domain requires nonempty final PEC geometry")
        allocation = self.allocation(fmin, fmax)
        h = allocation["spacing"]
        pml_cells, outer, inner, margin = allocation["cells"]
        n = pml_cells + outer + inner + margin
        low, high = np.array(bounds)[[0, 2]], np.array(bounds)[[1, 3]]
        offset = n * h - low
        size = tuple(high - low + 2 * n * h)
        if np.any(high <= low):
            raise ValueError("Final PEC geometry must have positive width and height")
        # Discard a design canvas, if supplied; only final material sets the domain.
        design = Geometry(None, geometry.shapes, geometry.next_id)
        resolved = design.translated(offset, size=size)
        collar = AxisCollar(pml_cells, pml_cells * h)
        c = (pml_cells + outer) * h
        t = c + inner * h
        origin = (
            (low + high) / 2 if self.phase_origin is None else np.asarray(self.phase_origin)
        ) + offset
        if not (t < origin[0] < size[0] - t and t < origin[1] < size[1] - t):
            raise ValueError("DomainPolicy.phase_origin must lie inside the derived TFSF box")
        layout = ScatteringLayout(
            (t, size[0] - t, t, size[1] - t),
            (c, size[0] - c, c, size[1] - c),
            (pml_cells + outer // 2) * h,
            tuple(origin),
            exterior_cells=(outer, inner),
            scatterer_margin_cells=margin,
            margin_mesh=self.margin_mesh,
        )
        return resolved, PML(collar, collar), layout, tuple(offset)
