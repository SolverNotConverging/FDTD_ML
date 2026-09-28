"""Reference-qualified, non-CNN mesh studies and procedural PEC shapes."""

import numpy as np

from ..api import DFTConvergence, Simulation, SolverSettings
from ..constants import C0
from .engineered_shapes import ENGINEERED_SHAPES, make_engineered_geometry
from .feasible import FeasibleSettings, analyze_mesh_adaptivity
from .optimize import Optimization, optimize_mesh
from .reference import Reference, ReferenceSettings, qualify_reference
from .shapes import SHAPES, make_geometry
from .study import plot_gallery, run_sweep


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
    sim = Simulation(
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
        sim.fit_domain(scatterer_margin_cells=scatterer_margin_cells)
    return sim


__all__ = [
    "ENGINEERED_SHAPES",
    "make_engineered_geometry",
    "SHAPES",
    "make_geometry",
    "make_simulation",
    "Reference",
    "ReferenceSettings",
    "qualify_reference",
    "Optimization",
    "optimize_mesh",
    "FeasibleSettings",
    "analyze_mesh_adaptivity",
    "run_sweep",
    "plot_gallery",
]
