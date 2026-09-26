"""Approximate bird silhouette with a lossy dielectric material."""


def make_scene():
    # One continuous outline: beak/body at right, wings above/below, tail at left.
    outline = [
        (0.20, 0.00), (0.12, 0.05), (0.06, 0.05), (0.02, 0.12),
        (-0.12, 0.35), (-0.20, 0.37), (-0.14, 0.10), (-0.22, 0.05),
        (-0.36, 0.10), (-0.40, 0.04), (-0.26, 0.00), (-0.40, -0.04),
        (-0.36, -0.10), (-0.22, -0.05), (-0.14, -0.10), (-0.20, -0.37),
        (-0.12, -0.35), (0.02, -0.12), (0.06, -0.05), (0.12, -0.05),
    ]
    return {
        "schema_version": 3,
        "lineage_id": "example_bird",
        "family": "bird_silhouette",
        "stage": "C8",
        "split": "example",
        "shape": "concave_polygon",
        "center_m": [0.6, 0.6],
        "vertices_m": [[0.6 + 0.5 * x, 0.6 + 0.5 * y] for x, y in outline],
        "feature_size_m": 0.025,
        "material": {"kind": "dielectric", "epsilon_r": 2.0, "sigma_e_s_per_m": 0.01},
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Bird silhouette")
