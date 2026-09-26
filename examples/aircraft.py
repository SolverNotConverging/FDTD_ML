"""Approximate top-view aircraft outline from the project's procedural shapes."""

from scattermesh.curriculum_v2 import _engineering_definition


def make_scene():
    material = {"kind": "dielectric", "epsilon_r": 4.0, "sigma_e_s_per_m": 0.01}
    return {
        "schema_version": 3,
        "lineage_id": "example_aircraft",
        "family": "aircraft_silhouette",
        "stage": "C8",
        "split": "example",
        **_engineering_definition(
            "aircraft", 1, center=(0.6, 0.6), extent=0.38,
            angle=0.12, material=material,
        ),
    }


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Aircraft silhouette")
