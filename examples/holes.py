"""CNN mesh demonstration for a dielectric plate with two through holes."""


def make_scene():
    return {
        "schema_version": 4,
        "lineage_id": "example_two_holes",
        "family": "example_polygon_with_holes",
        "stage": "C6",
        "split": "example",
        "objects": [
            {
                "object_id": "plate_with_two_holes",
                "shape": "polygon_with_holes",
                "center_m": [0.6, 0.6],
                "outer_ring_m": [
                    [0.4, 0.45],
                    [0.8, 0.45],
                    [0.8, 0.75],
                    [0.4, 0.75],
                ],
                "holes_m": [
                    [[0.45, 0.53], [0.54, 0.53], [0.54, 0.67], [0.45, 0.67]],
                    [[0.66, 0.53], [0.75, 0.53], [0.75, 0.67], [0.66, 0.67]],
                ],
                "feature_size_m": 0.09,
                "material": {
                    "kind": "dielectric",
                    "epsilon_r": 4.0,
                    "sigma_e_s_per_m": 0.0,
                },
            }
        ],
        "frequency_hz": 1.0e9,
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Plate with holes")
