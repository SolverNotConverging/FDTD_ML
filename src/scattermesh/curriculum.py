"""Deterministic scene pools and grouped task expansion for mesh learning."""

import hashlib
import json

import numpy as np

from .constants import C0

DOMAIN = 1.2
FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
PRIMARY_BUDGETS = (32, 48, 64, 96)
TRAIN_ANGLES = (0.2, 1.3, 3.0)
VALIDATION_ANGLES = (0.85, 4.2)
TEST_ANGLES = (2.1, 5.4)


def _identifier(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def simple_dielectric_pool():
    """Return 32 grouped circle geometries for the first analytic-reference pool."""
    lineages = (
        ("train", 0.045, 2.0, 0.00),
        ("train", 0.055, 2.5, 0.00),
        ("train", 0.065, 3.0, 0.01),
        ("train", 0.075, 3.5, 0.00),
        ("train", 0.085, 4.0, 0.03),
        ("train", 0.095, 4.5, 0.05),
        ("validation", 0.110, 3.25, 0.02),
        ("test", 0.125, 5.0, 0.04),
    )
    variants = (
        (0.94, (-0.036, 0.023)),
        (0.98, (0.031, -0.027)),
        (1.02, (-0.019, -0.034)),
        (1.06, (0.027, 0.032)),
    )
    pool = []
    for lineage_index, (split, base_radius, epsilon_r, sigma_e) in enumerate(lineages):
        lineage_id = f"simple_dk_lineage_{lineage_index:02d}"
        angles = {
            "train": TRAIN_ANGLES,
            "validation": VALIDATION_ANGLES,
            "test": TEST_ANGLES,
        }[split]
        for variant_index, (scale, offset) in enumerate(variants):
            radius = base_radius * scale
            center = (0.6 + offset[0], 0.6 + offset[1])
            definition = dict(
                family="simple",
                shape="circle",
                lineage_id=lineage_id,
                variant_index=variant_index,
                split=split,
                radius_m=radius,
                center_m=center,
                epsilon_r=epsilon_r,
                sigma_e_s_per_m=sigma_e,
            )
            feature_size = 2 * radius
            pool.append(
                dict(
                    geometry_id=f"simple_dk_{_identifier(definition)}",
                    **definition,
                    feature_size_m=feature_size,
                    feature_cells_on_256_input=feature_size / (DOMAIN / 256),
                    minimum_internal_wavelength_m=C0 / (max(FREQUENCIES) * np.sqrt(epsilon_r)),
                    incidence_angles_rad=list(angles),
                    frequencies_hz=list(FREQUENCIES),
                    analytic_reference="infinite_TM_z_dielectric_cylinder",
                )
            )
    return pool


def simple_candidate_tasks(pool=None, budgets=PRIMARY_BUDGETS):
    """Expand each geometry into illumination/budget conditions without splitting lineages."""
    pool = simple_dielectric_pool() if pool is None else list(pool)
    tasks = []
    for geometry in pool:
        for angle_index, angle in enumerate(geometry["incidence_angles_rad"]):
            illumination_id = f"{geometry['geometry_id']}_a{angle_index}"
            for cells in budgets:
                condition = dict(
                    geometry_id=geometry["geometry_id"],
                    lineage_id=geometry["lineage_id"],
                    split=geometry["split"],
                    family=geometry["family"],
                    illumination_id=illumination_id,
                    incidence_angle_rad=angle,
                    cells_x=cells,
                    cells_y=cells,
                )
                tasks.append(dict(task_id=f"condition_{_identifier(condition)}", **condition))
    return tasks
