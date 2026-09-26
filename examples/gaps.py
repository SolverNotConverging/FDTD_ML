"""CNN mesh demonstration for two dielectric circles with a narrow gap."""


def make_scene():
    material = {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0}
    radius = 0.09
    gap = 0.025
    offset = radius + gap / 2
    return {
        "schema_version": 3,
        "lineage_id": "example_gap_pair",
        "family": "example_two_circle_gap",
        "stage": "C4",
        "split": "example",
        "objects": [
            {
                "object_id": "left_circle",
                "shape": "circle",
                "center_m": [0.6 - offset, 0.6],
                "radius_m": radius,
                "feature_size_m": gap,
                "material": material.copy(),
            },
            {
                "object_id": "right_circle",
                "shape": "circle",
                "center_m": [0.6 + offset, 0.6],
                "radius_m": radius,
                "feature_size_m": gap,
                "material": material.copy(),
            },
        ],
        "gap_m": gap,
        "center_separation_m": 2 * radius + gap,
        "feature_size_m": gap,
        "frequency_hz": 1.0e9,
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Near gap")
