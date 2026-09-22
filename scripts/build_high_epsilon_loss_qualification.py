#!/usr/bin/env python3
"""Build a controlled high-permittivity, lossy-cylinder qualification sweep."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scattermesh.constants import C0, EPS0

DOMAIN = 1.2
REFERENCE_FREQUENCY_HZ = 1.0e9
FREQUENCIES_HZ = (0.8e9, 1.0e9, 1.2e9)
EPSILON_R_VALUES = (10.0, 20.0, 30.0)
RADII_M = (0.050, 0.120)
LOSS_TANGENTS = (0.03, 0.10, 0.20)
BUDGETS = (32, 96)
CANDIDATES = ("uniform", "region_strong", "hybrid_wide")


def identifier(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("configs/high_epsilon_loss_qualification_pool.json"),
    )
    parser.add_argument(
        "--campaign-output",
        type=Path,
        default=Path("configs/high_epsilon_loss_qualification.json"),
    )
    parser.add_argument("--duration-ns", type=float, default=70.0)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.duration_ns <= 0:
        raise ValueError("duration-ns must be positive")

    geometries = []
    conditions = []
    for epsilon_r in EPSILON_R_VALUES:
        for radius_m in RADII_M:
            for loss_tangent in LOSS_TANGENTS:
                sigma_e = (
                    loss_tangent
                    * 2
                    * np.pi
                    * REFERENCE_FREQUENCY_HZ
                    * EPS0
                    * epsilon_r
                )
                definition = {
                    "family": "simple",
                    "shape": "circle",
                    "split": "qualification",
                    "radius_m": radius_m,
                    "center_m": (0.617, 0.583),
                    "epsilon_r": epsilon_r,
                    "sigma_e_s_per_m": float(sigma_e),
                    "loss_tangent_at_1ghz": loss_tangent,
                }
                lineage_id = f"high_epsilon_loss_lineage_{identifier(definition)}"
                geometry_id = f"high_epsilon_loss_{identifier(definition)}"
                geometry = {
                    "geometry_id": geometry_id,
                    "lineage_id": lineage_id,
                    "variant_index": 0,
                    **definition,
                    "feature_size_m": 2 * radius_m,
                    "feature_cells_on_256_input": 2 * radius_m / (DOMAIN / 256),
                    "minimum_internal_wavelength_m": C0
                    / (max(FREQUENCIES_HZ) * np.sqrt(epsilon_r)),
                    "incidence_angles_rad": [0.85],
                    "frequencies_hz": list(FREQUENCIES_HZ),
                    "analytic_reference": "infinite_TM_z_dielectric_cylinder",
                }
                geometries.append(geometry)
                for cells in BUDGETS:
                    condition = {
                        "geometry_id": geometry_id,
                        "lineage_id": lineage_id,
                        "split": "qualification",
                        "family": "simple",
                        "illumination_id": f"{geometry_id}_a0",
                        "incidence_angle_rad": 0.85,
                        "cells_x": cells,
                        "cells_y": cells,
                    }
                    conditions.append(
                        {"task_id": f"condition_{identifier(condition)}", **condition}
                    )

    manifest = {
        "schema_version": 1,
        "purpose": "settling qualification for high-permittivity lossy cylinders",
        "geometry_count": len(geometries),
        "condition_count": len(conditions),
        "geometries": geometries,
        "conditions": conditions,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["dataset_id"] = "high_epsilon_loss_" + hashlib.sha256(canonical).hexdigest()[:16]

    campaign = {
        "schema_version": 3,
        "dataset_id": manifest["dataset_id"],
        "purpose": "70 ns high-permittivity loss-tangent qualification",
        "geometry_ids": [row["geometry_id"] for row in geometries],
        "condition_ids": [row["task_id"] for row in conditions],
        "candidate_names": list(CANDIDATES),
        "duration_s": args.duration_ns * 1e-9,
        "ranking": {"mode": "fixed_axis_soft_nt", "nt_cost_exponent": 0.1},
    }
    canonical = json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
    campaign["campaign_id"] = "high_epsilon_loss_qualification_" + hashlib.sha256(
        canonical
    ).hexdigest()[:16]

    atomic_json(args.manifest_output, manifest)
    atomic_json(args.campaign_output, campaign)
    case_count = len(conditions) * len(CANDIDATES)
    print(
        f"{manifest['dataset_id']}: {len(geometries)} geometries, "
        f"{len(conditions)} conditions; {campaign['campaign_id']}: {case_count} cases"
    )


if __name__ == "__main__":
    main()
