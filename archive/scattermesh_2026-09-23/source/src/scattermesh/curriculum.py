"""Deterministic scene pools and grouped task expansion for mesh learning."""

import hashlib
import json

import numpy as np

from .constants import C0, EPS0

DOMAIN = 1.2
FREQUENCIES = (0.8e9, 1.0e9, 1.2e9)
PRIMARY_BUDGETS = (32, 48, 64, 96)
TRAIN_ANGLES = (0.2, 1.3, 3.0)
VALIDATION_ANGLES = (0.85, 4.2)
TEST_ANGLES = (2.1, 5.4)
REMEDIATION_BUDGETS = (32, 48)
REMEDIATION_ANGLES = {
    "train": (0.10, 1.80, 3.40, 5.80),
    "validation": (0.65, 4.45),
    "test": (2.35, 5.15),
}
CORNER_REMEDIATION_ANGLES = {
    "train": (4.15, 4.55, 5.25, 5.65),
    "validation": (4.35, 5.45),
    "test": (4.70, 5.10),
}
LOSS_TANGENT_REFERENCE_FREQUENCY_HZ = 1.0e9


def _identifier(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _conductivity_for_loss_tangent(epsilon_r, loss_tangent):
    return (
        loss_tangent
        * 2
        * np.pi
        * LOSS_TANGENT_REFERENCE_FREQUENCY_HZ
        * EPS0
        * epsilon_r
    )


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


def factorial_simple_dielectric_pool():
    """Return a decorrelated, settling-qualified circle pool for the first CNN stage.

    The main training split is a Cartesian product of permittivity, radius, and
    positive conductivity, plus stable low-contrast lossless controls. Materials
    above epsilon_r=10 use loss tangent >=0.10 at 1 GHz, which settled through
    epsilon_r=30 in the 70 ns qualification sweep.
    """
    split_factors = {
        "train": (
            *(
                (epsilon_r, radius, sigma_e)
                for epsilon_r in (2.0, 4.0, 6.0, 8.0)
                for radius in (0.050, 0.085, 0.120)
                for sigma_e in (0.01, 0.06)
            ),
            *(
                (epsilon_r, radius, 0.0)
                for epsilon_r in (2.0, 4.0)
                for radius in (0.050, 0.120)
            ),
            *(
                (
                    epsilon_r,
                    radius,
                    _conductivity_for_loss_tangent(epsilon_r, loss_tangent),
                )
                for epsilon_r in (12.0, 20.0, 30.0)
                for radius in (0.050, 0.085, 0.120)
                for loss_tangent in (0.10, 0.20)
            ),
        ),
        "validation": (
            *(
                (epsilon_r, radius, sigma_e)
                for epsilon_r in (3.0, 7.0)
                for radius in (0.0675, 0.1025)
                for sigma_e in (0.02, 0.08)
            ),
            *(
                (
                    epsilon_r,
                    radius,
                    _conductivity_for_loss_tangent(epsilon_r, 0.14),
                )
                for epsilon_r in (16.0, 26.0)
                for radius in (0.0675, 0.1025)
            ),
        ),
        "test": (
            *(
                (epsilon_r, radius, sigma_e)
                for epsilon_r in (5.0, 10.0)
                for radius in (0.060, 0.110)
                for sigma_e in (0.01, 0.04)
            ),
            *(
                (
                    epsilon_r,
                    radius,
                    _conductivity_for_loss_tangent(epsilon_r, loss_tangent),
                )
                for epsilon_r in (14.0, 28.0)
                for radius in (0.060, 0.110)
                for loss_tangent in (0.12, 0.18)
            ),
        ),
    }
    split_angles = {
        "train": TRAIN_ANGLES,
        "validation": VALIDATION_ANGLES,
        "test": TEST_ANGLES,
    }
    pool = []
    lineage_index = 0
    for split, combinations in split_factors.items():
        for epsilon_r, base_radius, sigma_e in combinations:
            lineage = {
                "family": "simple",
                "shape": "circle",
                "split": split,
                "base_radius_m": base_radius,
                "epsilon_r": epsilon_r,
                "sigma_e_s_per_m": sigma_e,
                "loss_tangent_at_1ghz": sigma_e
                / (
                    2
                    * np.pi
                    * LOSS_TANGENT_REFERENCE_FREQUENCY_HZ
                    * EPS0
                    * epsilon_r
                ),
            }
            lineage_id = f"simple_factorial_lineage_{_identifier(lineage)}"
            orientation = 0.71 * lineage_index
            cosine, sine = np.cos(orientation), np.sin(orientation)
            for variant_index, (scale, raw_offset) in enumerate(
                ((0.97, (-0.041, 0.029)), (1.03, (0.037, -0.033)))
            ):
                dx = cosine * raw_offset[0] - sine * raw_offset[1]
                dy = sine * raw_offset[0] + cosine * raw_offset[1]
                radius = base_radius * scale
                definition = {
                    "family": "simple",
                    "shape": "circle",
                    "lineage_id": lineage_id,
                    "variant_index": variant_index,
                    "split": split,
                    "radius_m": radius,
                    "center_m": (0.6 + dx, 0.6 + dy),
                    "epsilon_r": epsilon_r,
                    "sigma_e_s_per_m": sigma_e,
                    "loss_tangent_at_1ghz": lineage["loss_tangent_at_1ghz"],
                }
                feature_size = 2 * radius
                pool.append(
                    {
                        "geometry_id": f"simple_factorial_{_identifier(definition)}",
                        **definition,
                        "feature_size_m": feature_size,
                        "feature_cells_on_256_input": feature_size / (DOMAIN / 256),
                        "minimum_internal_wavelength_m": C0
                        / (max(FREQUENCIES) * np.sqrt(epsilon_r)),
                        "incidence_angles_rad": list(split_angles[split]),
                        "frequencies_hz": list(FREQUENCIES),
                        "analytic_reference": "infinite_TM_z_dielectric_cylinder",
                    }
                )
            lineage_index += 1
    return pool


def circle_remediation_pool():
    """Return disjoint translated/scaled circles for the failed frozen OOD gate.

    None of the radii, centers, material tuples, or incidence angles duplicates the
    frozen position/scale suite.  Two nearby variants remain grouped by lineage.
    """
    lineages = (
        ("train", 0.035, (0.28, 0.31), 4.0, 0.00),
        ("train", 0.035, (0.89, 0.31), 18.0, 0.12),
        ("train", 0.035, (0.31, 0.89), 7.0, 0.05),
        ("train", 0.035, (0.89, 0.88), 24.0, 0.16),
        ("train", 0.130, (0.40, 0.41), 4.0, 0.00),
        ("train", 0.130, (0.80, 0.40), 18.0, 0.12),
        ("train", 0.130, (0.41, 0.80), 7.0, 0.05),
        ("train", 0.130, (0.79, 0.79), 24.0, 0.16),
        ("train", 0.145, (0.43, 0.44), 4.0, 0.02),
        ("train", 0.145, (0.77, 0.43), 18.0, 0.12),
        ("train", 0.145, (0.44, 0.77), 7.0, 0.05),
        ("train", 0.145, (0.76, 0.76), 24.0, 0.16),
        ("validation", 0.045, (0.30, 0.62), 5.0, 0.03),
        ("validation", 0.115, (0.75, 0.48), 16.0, 0.13),
        ("validation", 0.140, (0.48, 0.75), 8.0, 0.06),
        ("validation", 0.140, (0.74, 0.74), 22.0, 0.15),
        ("test", 0.055, (0.90, 0.58), 6.0, 0.04),
        ("test", 0.120, (0.45, 0.76), 14.0, 0.11),
        ("test", 0.142, (0.76, 0.45), 10.0, 0.07),
        ("test", 0.142, (0.75, 0.75), 26.0, 0.17),
    )
    variants = ((0.985, (-0.007, 0.005)), (1.015, (0.006, -0.007)))
    pool = []
    for lineage_index, (split, base_radius, base_center, epsilon_r, loss_tangent) in enumerate(
        lineages
    ):
        lineage = {
            "family": "simple_remediation",
            "shape": "circle",
            "split": split,
            "base_radius_m": base_radius,
            "base_center_m": base_center,
            "epsilon_r": epsilon_r,
            "loss_tangent_at_1ghz": loss_tangent,
        }
        lineage_id = f"circle_remediation_lineage_{_identifier(lineage)}"
        orientation = 0.47 * lineage_index
        cosine, sine = np.cos(orientation), np.sin(orientation)
        for variant_index, (scale, raw_offset) in enumerate(variants):
            dx = cosine * raw_offset[0] - sine * raw_offset[1]
            dy = sine * raw_offset[0] + cosine * raw_offset[1]
            radius = base_radius * scale
            center = (base_center[0] + dx, base_center[1] + dy)
            definition = {
                "family": "simple_remediation",
                "shape": "circle",
                "lineage_id": lineage_id,
                "variant_index": variant_index,
                "split": split,
                "radius_m": radius,
                "center_m": center,
                "epsilon_r": epsilon_r,
                "loss_tangent_at_1ghz": loss_tangent,
                "sigma_e_s_per_m": _conductivity_for_loss_tangent(epsilon_r, loss_tangent),
            }
            feature_size = 2 * radius
            pool.append(
                {
                    "geometry_id": f"circle_remediation_{_identifier(definition)}",
                    **definition,
                    "feature_size_m": feature_size,
                    "feature_cells_on_256_input": feature_size / (DOMAIN / 256),
                    "minimum_internal_wavelength_m": C0
                    / (max(FREQUENCIES) * np.sqrt(epsilon_r)),
                    "incidence_angles_rad": list(REMEDIATION_ANGLES[split]),
                    "frequencies_hz": list(FREQUENCIES),
                    "analytic_reference": "infinite_TM_z_dielectric_cylinder",
                }
            )
    return pool


def circle_corner_remediation_pool():
    """Dense, split-disjoint coverage around the remaining lossy corner failure regime."""
    corner_centers = {
        "northeast": (0.79, 0.79),
        "northwest": (0.41, 0.79),
        "southeast": (0.79, 0.41),
        "southwest": (0.41, 0.41),
    }
    train_factors = (
        (0.124, 17.0, 0.11),
        (0.136, 21.0, 0.15),
        (0.148, 23.0, 0.17),
    )
    lineages = []
    for corner_index, (corner, center) in enumerate(corner_centers.items()):
        for factor_index, (radius, epsilon_r, loss_tangent) in enumerate(train_factors):
            inward = 0.008 * factor_index
            x_sign = -1 if center[0] > DOMAIN / 2 else 1
            y_sign = -1 if center[1] > DOMAIN / 2 else 1
            shifted = (center[0] + x_sign * inward, center[1] + y_sign * inward)
            lineages.append(
                ("train", corner, radius, shifted, epsilon_r, loss_tangent, corner_index)
            )
    for split, radius, epsilon_r, loss_tangent, inward in (
        ("validation", 0.131, 18.0, 0.12, 0.014),
        ("test", 0.142, 22.0, 0.16, 0.026),
    ):
        for corner_index, (corner, center) in enumerate(corner_centers.items()):
            x_sign = -1 if center[0] > DOMAIN / 2 else 1
            y_sign = -1 if center[1] > DOMAIN / 2 else 1
            shifted = (center[0] + x_sign * inward, center[1] + y_sign * inward)
            lineages.append(
                (split, corner, radius, shifted, epsilon_r, loss_tangent, corner_index)
            )

    variants = ((0.98, (-0.006, 0.004)), (1.02, (0.005, -0.006)))
    pool = []
    for lineage_index, (
        split,
        corner,
        base_radius,
        base_center,
        epsilon_r,
        loss_tangent,
        corner_index,
    ) in enumerate(lineages):
        lineage = {
            "family": "simple_corner_remediation",
            "shape": "circle",
            "split": split,
            "corner": corner,
            "base_radius_m": base_radius,
            "base_center_m": base_center,
            "epsilon_r": epsilon_r,
            "loss_tangent_at_1ghz": loss_tangent,
        }
        lineage_id = f"circle_corner_remediation_lineage_{_identifier(lineage)}"
        orientation = 0.39 * lineage_index + 0.17 * corner_index
        cosine, sine = np.cos(orientation), np.sin(orientation)
        for variant_index, (scale, raw_offset) in enumerate(variants):
            dx = cosine * raw_offset[0] - sine * raw_offset[1]
            dy = sine * raw_offset[0] + cosine * raw_offset[1]
            radius = base_radius * scale
            center = (base_center[0] + dx, base_center[1] + dy)
            definition = {
                "family": "simple_corner_remediation",
                "shape": "circle",
                "lineage_id": lineage_id,
                "variant_index": variant_index,
                "split": split,
                "corner": corner,
                "radius_m": radius,
                "center_m": center,
                "epsilon_r": epsilon_r,
                "loss_tangent_at_1ghz": loss_tangent,
                "sigma_e_s_per_m": _conductivity_for_loss_tangent(epsilon_r, loss_tangent),
            }
            feature_size = 2 * radius
            pool.append(
                {
                    "geometry_id": f"circle_corner_remediation_{_identifier(definition)}",
                    **definition,
                    "feature_size_m": feature_size,
                    "feature_cells_on_256_input": feature_size / (DOMAIN / 256),
                    "minimum_internal_wavelength_m": C0
                    / (max(FREQUENCIES) * np.sqrt(epsilon_r)),
                    "incidence_angles_rad": list(CORNER_REMEDIATION_ANGLES[split]),
                    "frequencies_hz": list(FREQUENCIES),
                    "analytic_reference": "infinite_TM_z_dielectric_cylinder",
                }
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
