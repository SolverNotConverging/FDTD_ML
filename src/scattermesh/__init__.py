"""Scattered-field TMz solver; explicit NumPy reference backend for qualification."""

from .geometry import PEC, Circle, Material, Rectangle
from .grid import Grid, focused_axis
from .meshing import circular_interface_axes, density_axis
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
    "density_axis",
    "circular_interface_axes",
    "PlaneWave",
    "SimulationResult",
    "simulate",
    "simulate_cuda",
]
