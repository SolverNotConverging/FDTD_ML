"""Qualified numerical references and fixed-budget mesh optimization."""

from .feasible import FeasibleSettings, analyze_mesh_adaptivity
from .optimize import Optimization, SearchSettings, optimize_mesh
from .reference import Reference, ReferenceSettings, qualify_reference

__all__ = [
    "Reference",
    "ReferenceSettings",
    "qualify_reference",
    "Optimization",
    "SearchSettings",
    "optimize_mesh",
    "FeasibleSettings",
    "analyze_mesh_adaptivity",
]
