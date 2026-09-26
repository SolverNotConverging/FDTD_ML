"""Versioned topology-scene examples for C6 development and smoke checks."""

import math


def square_ring_smoke_scene(
    *, material_kind="dielectric", wall_thickness_m=0.09, sigma_e_s_per_m=0.0
):
    """Return a small schema-4 ring scene with one square void."""
    if material_kind not in {"dielectric", "pec"}:
        raise ValueError("Ring material kind must be dielectric or pec")
    if not 0 < wall_thickness_m < 0.15:
        raise ValueError("Square-ring wall thickness must lie between zero and 0.15 m")
    if (
        isinstance(sigma_e_s_per_m, bool)
        or not isinstance(sigma_e_s_per_m, (int, float))
        or not math.isfinite(sigma_e_s_per_m)
        or sigma_e_s_per_m < 0
    ):
        raise ValueError("Conductivity must be a finite nonnegative number")
    inner_low, inner_high = 0.45 + wall_thickness_m, 0.75 - wall_thickness_m
    material = (
        {"kind": "pec"}
        if material_kind == "pec"
        else {
            "kind": "dielectric",
            "epsilon_r": 4.0,
            "sigma_e_s_per_m": float(sigma_e_s_per_m),
        }
    )
    thickness_mm = int(round(1000 * wall_thickness_m))
    lineage = (
        "c6_square_ring_smoke" if thickness_mm == 90 else f"c6_square_ring_t{thickness_mm:03d}mm"
    )
    return {
        "schema_version": 4,
        "lineage_id": lineage,
        "family": "topology_ring",
        "stage": "C6",
        "split": "development",
        "ring_wall_thickness_m": float(wall_thickness_m),
        "objects": [
            {
                "object_id": "square_ring",
                "shape": "polygon_with_holes",
                "center_m": [0.6, 0.6],
                "outer_ring_m": [
                    [0.45, 0.45],
                    [0.75, 0.45],
                    [0.75, 0.75],
                    [0.45, 0.75],
                ],
                "holes_m": [
                    [
                        [inner_low, inner_low],
                        [inner_high, inner_low],
                        [inner_high, inner_high],
                        [inner_low, inner_high],
                    ]
                ],
                "feature_size_m": 0.2,
                "material": material,
            }
        ],
        "frequency_hz": 1e9,
    }


def open_cavity_smoke_scene(*, sigma_e_s_per_m=0.0):
    """Return a schema-4 concave U-shape with a cavity open to free space."""
    return {
        "schema_version": 4,
        "lineage_id": "c6_open_cavity_smoke",
        "family": "topology_open_cavity",
        "stage": "C6",
        "split": "development",
        "objects": [
            {
                "object_id": "u_cavity",
                "shape": "concave_polygon",
                "center_m": [0.6, 0.6],
                "vertices_m": [
                    [0.4, 0.4],
                    [0.8, 0.4],
                    [0.8, 0.8],
                    [0.68, 0.8],
                    [0.68, 0.52],
                    [0.52, 0.52],
                    [0.52, 0.8],
                    [0.4, 0.8],
                ],
                "feature_size_m": 0.12,
                "material": {
                    "kind": "dielectric",
                    "epsilon_r": 4.0,
                    "sigma_e_s_per_m": float(sigma_e_s_per_m),
                },
            }
        ],
        "frequency_hz": 1e9,
    }
