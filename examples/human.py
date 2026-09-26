"""Approximate standing human silhouette in the 2D analysis plane."""


def make_scene():
    outline = [
        (0.00, 0.42), (0.07, 0.40), (0.10, 0.34), (0.08, 0.28),
        (0.06, 0.25), (0.15, 0.22), (0.23, 0.13), (0.28, -0.01),
        (0.24, -0.04), (0.18, 0.08), (0.14, 0.12), (0.13, -0.08),
        (0.10, -0.38), (0.03, -0.38), (0.00, -0.15), (-0.03, -0.38),
        (-0.10, -0.38), (-0.13, -0.08), (-0.14, 0.12), (-0.18, 0.08),
        (-0.24, -0.04), (-0.28, -0.01), (-0.23, 0.13), (-0.15, 0.22),
        (-0.06, 0.25), (-0.08, 0.28), (-0.10, 0.34), (-0.07, 0.40),
    ]
    return {
        "schema_version": 3,
        "lineage_id": "example_human",
        "family": "human_silhouette",
        "stage": "C8",
        "split": "example",
        "shape": "concave_polygon",
        "center_m": [0.6, 0.6],
        "vertices_m": [[0.6 + 0.6 * x, 0.6 + 0.6 * y] for x, y in outline],
        "feature_size_m": 0.03,
        "material": {"kind": "dielectric", "epsilon_r": 2.0, "sigma_e_s_per_m": 0.02},
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Human silhouette")
