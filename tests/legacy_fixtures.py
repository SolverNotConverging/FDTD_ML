"""Frozen pre-cleanup layouts for numerical regression tests only."""

import numpy as np

from fdtdmesh.api import DFTConvergence, ScatteringLayout, Simulation, SolverSettings
from fdtdmesh.catalog.shapes import make_geometry
from fdtdmesh.constants import C0
from fdtdmesh.mesh import AxisCollar
from fdtdmesh.pml import PML


def make_simulation(
    shape="circle",
    *,
    incidence_deg=0.0,
    frequency=1e9,
    scale=None,
    scale_factor=None,
    settings=None,
    fit_domain=True,
    scatterer_margin_cells=5,
):
    """Rotated-object incidence experiment; SI geometry and radians for result angles.

    Add incidence_deg (in radians) to stored solver angles to report object-frame
    angles. Each incidence has independent reference and optimized meshes.

    scale is metres per recipe unit. Alternatively, scale_factor multiplies the
    default 0.6-wavelength recipe scale. Supplying both is an error.
    """
    if not np.isfinite([frequency, incidence_deg]).all() or frequency <= 0:
        raise ValueError("Frequency must be positive and incidence finite")
    lam = C0 / frequency
    if scale is not None and scale_factor is not None:
        raise ValueError("Supply scale (metres) or scale_factor (dimensionless), not both")
    if scale_factor is not None and (not np.isfinite(scale_factor) or scale_factor <= 0):
        raise ValueError("scale_factor must be positive and finite")
    if scale is None:
        scale = 0.6 * lam * (1.0 if scale_factor is None else scale_factor)
    settings = settings or SolverSettings(stop=DFTConvergence(check_interval=256))
    construction_size = max(6 * lam, 8 * float(scale)) if fit_domain else 6 * lam
    sim = fixed_simulation(
        (construction_size, construction_size),
        0.9 * frequency,
        1.1 * frequency,
        settings=settings,
        angles=np.deg2rad(np.arange(360) - incidence_deg),
    )
    sim.set_geometry(
        make_geometry(
            shape,
            size=sim.size,
            scale=scale,
            incidence_deg=incidence_deg,
        )
    )
    if fit_domain:
        fit_legacy_domain(sim, scatterer_margin_cells=scatterer_margin_cells)
    return sim


def fit_legacy_domain(sim, *, scatterer_margin_cells=5, exterior_cells=(6, 4)):
    """Fit the domain to final PEC bounds with fixed cell margins; invalidate mesh.

    Translate the entire exact recipe and phase origin together. Physical
    gap widths use the adjacent PML cell widths; PML thickness is unchanged.
    Call after adding geometry and before applying a mesh.
    """
    from fdtdmesh.mesh import cell_count

    margin = cell_count(scatterer_margin_cells)
    gaps = tuple(cell_count(n) for n in exterior_cells)
    if margin < 3 or len(gaps) != 2 or min(gaps) < 2:
        raise ValueError("Need >=3 scatterer-margin cells and >=2 cells in each exterior gap")
    bounds = sim.geometry.bounds
    if bounds is None:
        raise ValueError("Cannot fit a domain without PEC material")
    h = np.array(
        [p.thickness / p.cells if p.cells else sim.wavelength / 24 for p in (sim.pml.x, sim.pml.y)]
    )
    thickness = np.array([sim.pml.x.thickness, sim.pml.y.thickness])
    padding = thickness + (sum(gaps) + margin) * h
    low, high = np.array(bounds)[[0, 2]], np.array(bounds)[[1, 3]]
    offset = padding - low
    size = high - low + 2 * padding
    geometry = sim.geometry.translated(offset, size=tuple(size))
    t = thickness + sum(gaps) * h
    c = thickness + gaps[0] * h
    layout = ScatteringLayout(
        (t[0], size[0] - t[0], t[1], size[1] - t[1]),
        (c[0], size[0] - c[0], c[1], size[1] - c[1]),
        thickness[0] + (gaps[0] // 2) * h[0],
        tuple(np.asarray(sim.layout.origin) + offset),
        exterior_cells=gaps,
        scatterer_margin_cells=margin,
    )
    sim._geometry, sim._layout = geometry, layout
    sim._invalidate()
    return sim


def fixed_simulation(size, fmin, fmax, *, pml=None, layout=None, **kwargs):
    """Original manual-layout fixture. Not a supported production API."""
    lam = C0 / ((fmin + fmax) / 2)
    collar = AxisCollar(12, 0.5 * lam)
    pml = pml or PML(collar, collar)
    lx, ly = size
    px, py = pml.x.thickness, pml.y.thickness
    hx = px / pml.x.cells if pml.x.cells else lam / 24
    hy = py / pml.y.cells if pml.y.cells else lam / 24
    layout = layout or ScatteringLayout(
        (px + 10 * hx, lx - px - 10 * hx, py + 10 * hy, ly - py - 10 * hy),
        (px + 6 * hx, lx - px - 6 * hx, py + 6 * hy, ly - py - 6 * hy),
        px + 3 * hx,
        (lx / 2, ly / 2),
        exterior_cells=(6, 4),
    )
    return Simulation._from_resolved(size, fmin, fmax, pml=pml, layout=layout, **kwargs)
