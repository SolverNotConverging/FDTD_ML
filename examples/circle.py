"""CNN mesh demonstration for a dielectric circle."""


def make_scene():
    return {
        "schema_version": 2,
        "lineage_id": "example_circle",
        "family": "circle",
        "stage": "C0",
        "split": "example",
        "shape": "circle",
        "center_m": [0.6, 0.6],
        "radius_m": 0.12,
        "material": {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.0},
        "frequency_hz": 1.0e9,
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Circle")
