"""Geometry-first native-CUDA TMz PEC scattering."""

from .api import DFTConvergence, Simulation, SolverSettings
from .boundary import BoundaryPolicy, StaircaseFallbackWarning
from .constants import C0
from .domain import DomainPolicy
from .geometry import Geometry
from .mesh import AxisConstraints, Mesh
from .meshing import MeshOptions
from .result import Result
from .simulation import ConvergenceError

__all__ = [
    "Simulation",
    "DomainPolicy",
    "BoundaryPolicy",
    "SolverSettings",
    "DFTConvergence",
    "Geometry",
    "Mesh",
    "MeshOptions",
    "AxisConstraints",
    "Result",
    "C0",
    "ConvergenceError",
    "StaircaseFallbackWarning",
]
