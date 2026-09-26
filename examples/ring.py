"""CNN mesh demonstration for a square dielectric ring."""

from scattermesh.topology_v2 import square_ring_smoke_scene


def make_scene():
    scene = square_ring_smoke_scene(material_kind="dielectric", wall_thickness_m=0.075)
    scene["lineage_id"] = "example_square_ring"
    scene["family"] = "example_square_ring"
    scene["split"] = "example"
    return scene


if __name__ == "__main__":
    from common import run_example

    run_example(make_scene(), title="Square ring")
