"""CNN mesh demonstration for a rotated dielectric ellipse."""


def make_scene():
    return {
        "schema_version": 2,
        "lineage_id": "example_ellipse",
        "family": "ellipse",
        "stage": "C1",
        "split": "example",
        "shape": "ellipse",
        "center_m": [0.6, 0.6],
        "radii_m": [0.16, 0.08],
        "angle_rad": 0.3,
        "material": {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0},
        "frequency_hz": 1.0e9,
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Ellipse")
