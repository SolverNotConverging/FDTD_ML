"""Scattered-field TMz solver; explicit NumPy reference backend for qualification."""

from .curriculum import (
    circle_corner_remediation_pool,
    circle_remediation_pool,
    factorial_simple_dielectric_pool,
    simple_candidate_tasks,
    simple_dielectric_pool,
)
from .distillation import (
    axis_probability,
    build_distillation_dataset,
    conditioning_features,
    conditioning_features_v2,
    merge_distillation_datasets,
    probability_axis,
    rasterize_circle,
    rasterize_scene,
    resample_axis_profiles,
)
from .geometry import PEC, Circle, Material, Rectangle
from .grid import Grid, focused_axis
from .meshing import circular_interface_axes, density_axis
from .solver import SimulationResult, simulate
from .solver_cuda import simulate_cuda
from .source import PlaneWave
from .sparse import (
    circle_cluster_objects,
    sparse_candidate_axes,
    sparse_circle_candidate_axes,
    sparse_cluster_metrics,
    sparse_scene_objects,
)

__all__ = [
    "PEC",
    "Circle",
    "Material",
    "Rectangle",
    "Grid",
    "focused_axis",
    "density_axis",
    "circular_interface_axes",
    "simple_dielectric_pool",
    "circle_remediation_pool",
    "circle_corner_remediation_pool",
    "factorial_simple_dielectric_pool",
    "simple_candidate_tasks",
    "axis_probability",
    "probability_axis",
    "rasterize_circle",
    "rasterize_scene",
    "conditioning_features",
    "conditioning_features_v2",
    "resample_axis_profiles",
    "build_distillation_dataset",
    "merge_distillation_datasets",
    "PlaneWave",
    "SimulationResult",
    "simulate",
    "simulate_cuda",
    "circle_cluster_objects",
    "sparse_scene_objects",
    "sparse_cluster_metrics",
    "sparse_candidate_axes",
    "sparse_circle_candidate_axes",
]
