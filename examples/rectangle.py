"""CNN mesh demonstration for a dielectric rectangle."""


def make_scene():
    return {
        "schema_version": 2,
        "lineage_id": "example_rectangle",
        "family": "rectangle",
        "stage": "C0",
        "split": "example",
        "shape": "rectangle",
        "center_m": [0.6, 0.6],
        "bounds_m": [0.45, 0.75, 0.50, 0.70],
        "material": {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0},
        "frequency_hz": 1.0e9,
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Rectangle")
