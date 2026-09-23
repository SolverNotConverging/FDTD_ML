"""Numerical TMz scattering interfaces for the mesh-CNN v2 project."""

from .geometry import PEC, Circle, Material, Rectangle
from .geometry_v2 import Ellipse, Polygon, SmoothLobed, oriented_rectangle
from .grid import Grid, focused_axis
from .solver import SimulationResult, simulate
from .solver_cuda import simulate_cuda
from .source import PlaneWave

__all__ = [
    "PEC",
    "Circle",
    "Material",
    "Rectangle",
    "Ellipse",
    "Polygon",
    "SmoothLobed",
    "oriented_rectangle",
    "Grid",
    "focused_axis",
    "PlaneWave",
    "SimulationResult",
    "simulate",
    "simulate_cuda",
]
