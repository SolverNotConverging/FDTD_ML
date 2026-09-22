"""Frozen analytic-circle cases for position and scale generalization."""

import hashlib
import json

import numpy as np

from .constants import EPS0

DOMAIN = 1.2
# The translated r=0.135 m cases need a legal NF2FF contour on the exact 32-cell
# grid.  A 0.15 m PML leaves enough geometric clearance but not enough mesh nodes
# for both the contour interpolation stencil and one full vacuum buffer cell.
PML_THICKNESS = 0.12
REFERENCE_FREQUENCY_HZ = 1.0e9
FREQUENCIES_HZ = (0.8e9, 1.0e9, 1.2e9)
GENERALIZATION_BUDGETS = (32, 48)
GENERALIZATION_ANGLES = (0.35, 4.9)
GENERALIZATION_CENTERS = {
    "center": (0.60, 0.60),
    "southwest": (0.38, 0.38),
    "southeast": (0.82, 0.38),
    "northwest": (0.38, 0.82),
    "northeast": (0.82, 0.82),
}
GENERALIZATION_RADII = {
    "below_training_range": 0.040,
    "interpolation": 0.075,
    "above_training_range": 0.135,
}
GENERALIZATION_MATERIALS = {
    "low_lossless": (4.0, 0.0),
    "high_lossy": (20.0, 0.14),
}


def _identifier(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _conductivity(epsilon_r, loss_tangent):
    return loss_tangent * 2 * np.pi * REFERENCE_FREQUENCY_HZ * EPS0 * epsilon_r


def circle_generalization_examples():
    """Return the result-blind frozen position/scale extrapolation suite."""
    examples = []
    for size_regime, radius in GENERALIZATION_RADII.items():
        for position_id, center in GENERALIZATION_CENTERS.items():
            for material_id, (epsilon_r, loss_tangent) in GENERALIZATION_MATERIALS.items():
                for angle_index, angle in enumerate(GENERALIZATION_ANGLES):
                    for cells in GENERALIZATION_BUDGETS:
                        definition = {
                            "size_regime": size_regime,
                            "radius_m": radius,
                            "position_id": position_id,
                            "center_m": center,
                            "material_id": material_id,
                            "epsilon_r": epsilon_r,
                            "loss_tangent_at_1ghz": loss_tangent,
                            "incidence_angle_rad": angle,
                            "cells_x": cells,
                            "cells_y": cells,
                        }
                        examples.append(
                            {
                                "sample_id": f"circle_generalization_{_identifier(definition)}",
                                "split": "generalization",
                                "family": "simple_ood",
                                "shape": "circle",
                                **definition,
                                "angle_index": angle_index,
                                "feature_size_m": 2 * radius,
                                "sigma_e_s_per_m": _conductivity(epsilon_r, loss_tangent),
                                "frequencies_hz": list(FREQUENCIES_HZ),
                            }
                        )
    return examples


def circle_generalization_plan():
    examples = circle_generalization_examples()
    identity = {
        "examples": examples,
        "ranking": {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1},
        "duration_schedule_s": [70e-9, 140e-9, 560e-9],
        "max_grading_ratio": 3.0,
        "pml_thickness_m": PML_THICKNESS,
        "monitor_policy": "widest_non_pml_enclosing",
        "gate": {
            "minimum_meaningful_win_fraction": 0.70,
            "minimum_each_size_regime_win_fraction": 0.50,
            "minimum_each_position_median_improvement": 1.0,
            "minimum_overall_median_improvement": 1.2,
            "minimum_worst_case_improvement": 0.5,
            "require_all_cases_settled": True,
        },
    }
    return {
        "schema_version": 1,
        "evaluation_id": f"circle_position_scale_{_identifier(identity)}",
        "purpose": "frozen analytic-circle position and scale generalization",
        **identity,
    }
