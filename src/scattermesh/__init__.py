"""Scattered-field TMz solver; explicit NumPy reference backend for qualification."""

from .geometry import PEC, Circle, Material, Rectangle
from .grid import Grid, focused_axis
from .solver import SimulationResult, simulate
from .solver_cuda import simulate_cuda
from .source import PlaneWave

__all__ = [
    "PEC",
    "Circle",
    "Material",
    "Rectangle",
    "Grid",
    "focused_axis",
    "PlaneWave",
    "SimulationResult",
    "simulate",
    "simulate_cuda",
]
