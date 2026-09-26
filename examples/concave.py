"""CNN mesh demonstration for an open dielectric cavity."""

from scattermesh.topology_v2 import open_cavity_smoke_scene


def make_scene():
    scene = open_cavity_smoke_scene()
    scene["lineage_id"] = "example_open_cavity"
    scene["family"] = "example_open_cavity"
    scene["split"] = "example"
    return scene


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Concave open cavity")
