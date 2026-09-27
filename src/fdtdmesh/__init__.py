"""Conformal enlarged-cell TMz scattering on nonuniform Cartesian grids."""

from .api import DFTConvergence, ScatteringLayout, Simulation, SolverSettings
from .constants import C0
from .geometry import Geometry
from .mesh import AxisCollar, AxisConstraints, Mesh, density_mesh
from .result import Result
from .scene import Scene2D
from .simulation import Convergence, ConvergenceError, ScatteringCase, run_scattering

__all__ = [
    "Simulation",
    "SolverSettings",
    "DFTConvergence",
    "ScatteringLayout",
    "Result",
    "Geometry",
    "C0",
    "ConvergenceError",
    "AxisCollar",
    "AxisConstraints",
    "Mesh",
    "Scene2D",
    "density_mesh",
    "Convergence",
    "ScatteringCase",
    "run_scattering",
]
