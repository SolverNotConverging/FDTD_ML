"""Approximate top-view warship outline from the project's procedural shapes."""

from scattermesh.curriculum_v2 import _engineering_definition


def make_scene():
    material = {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.01}
    return {
        "schema_version": 3,
        "lineage_id": "example_warship",
        "family": "warship_silhouette",
        "stage": "C8",
        "split": "example",
        **_engineering_definition(
            "ship", 2, center=(0.6, 0.6), extent=0.40,
            angle=-0.25, material=material,
        ),
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Warship silhouette")
