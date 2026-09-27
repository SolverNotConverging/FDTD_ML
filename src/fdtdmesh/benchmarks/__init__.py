"""Reference-qualified, non-CNN mesh studies and procedural PEC shapes."""

import numpy as np

from ..api import DFTConvergence, Simulation, SolverSettings
from ..constants import C0
from .optimize import Optimization, optimize_mesh
from .reference import Reference, ReferenceSettings, qualify_reference
from .shapes import SHAPES, make_geometry
from .study import plot_gallery, run_sweep


def make_simulation(shape="circle", *, incidence_deg=0.0, frequency=1e9, scale=None, settings=None):
    """Rotated-object incidence experiment; SI geometry and radians for result angles.

    Add incidence_deg (in radians) to stored solver angles to report object-frame
    angles. Each incidence has independent reference and optimized meshes.
    """
    if not np.isfinite([frequency, incidence_deg]).all() or frequency <= 0:
        raise ValueError("Frequency must be positive and incidence finite")
    lam = C0 / frequency
    settings = settings or SolverSettings(stop=DFTConvergence(check_interval=256))
    sim = Simulation(
        (6 * lam, 6 * lam),
        0.9 * frequency,
        1.1 * frequency,
        settings=settings,
        angles=np.deg2rad(np.arange(360) - incidence_deg),
    )
    sim.set_geometry(
        make_geometry(
            shape,
            size=sim.size,
            scale=0.6 * lam if scale is None else scale,
            incidence_deg=incidence_deg,
        )
    )
    return sim


__all__ = [
    "SHAPES",
    "make_geometry",
    "make_simulation",
    "Reference",
    "ReferenceSettings",
    "qualify_reference",
    "Optimization",
    "optimize_mesh",
    "run_sweep",
    "plot_gallery",
]
