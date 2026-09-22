from dataclasses import replace

import numpy as np
import pytest

from fdtdmesh.data.generate_v6 import RichConfig, make_rich_scene
from fdtdmesh.data.schema import SceneSpec


def spec(geometry):
    return SceneSpec(
        schema_version=1,
        scene_id="anchor-contract",
        group_id="test",
        seed=0,
        split="train",
        family="test",
        domain=[1.0, 1.0],
        f_min=1e7,
        f_max=1e8,
        t_end=1e-7,
        dtype="float64",
        raster_shape=[128, 128],
        budgets=[[128, 128]],
        pml=dict(pml_width=8, thickness=[0.125, 0.125]),
        materials=[],
        geometry=geometry,
        sources=[dict(kind="point", x=0.1875, y=0.1875, normalization="current")],
        receivers=[dict(kind="point", x=0.8125, y=0.8125)],
        anchor_probes=True,
        pec_policy="rectangles_and_wires",
    )


def test_exact_four_rectangle_and_three_wire_coordinates():
    empty = spec([]).build([128, 128], reference=True)
    rectangle = dict(
        kind="rectangle", material="PEC", x_position=[0.25, 0.75], y_position=[0.375, 0.625]
    )
    wire = dict(kind="pec_line", x=0.5, y=[0.25, 0.75])
    for geometry, expected_x, expected_y in [
        ([rectangle], {0.25, 0.75}, {0.375, 0.625}),
        ([wire], {0.5}, {0.25, 0.75}),
    ]:
        scene = spec(geometry)
        for n in (128, 256, 1024):
            sim = scene.build([n, n], reference=True)
            assert sim.x_anchors - empty.x_anchors == expected_x
            assert sim.y_anchors - empty.y_anchors == expected_y
            mesh = sim.mesh_uniform()
            assert all(x in mesh.x for x in sim.x_anchors)
            assert all(y in mesh.y for y in sim.y_anchors)


def test_reject_nonrectangular_pec_and_later_dielectric_cutouts():
    with pytest.raises(ValueError, match="only axis-aligned"):
        spec([dict(kind="circle", material="PEC", center=[0.5, 0.5], radius=0.2)]).validate()
    rectangle = dict(
        kind="rectangle", material="PEC", x_position=[0.25, 0.75], y_position=[0.25, 0.75]
    )
    hole = dict(kind="circle", material="vacuum", center=[0.5, 0.5], radius=0.15)
    with pytest.raises(ValueError, match="follow dielectrics"):
        spec([rectangle, hole]).validate()
    allowed = spec([hole, rectangle])
    allowed.validate()
    sim = allowed.build([128, 128])
    assert sim.sample([0.5], [0.5])[-1][0, 0]
    legacy = replace(allowed, pec_policy="legacy")
    assert "pec_policy" not in legacy.to_dict()
    assert SceneSpec.from_dict(allowed.to_dict()).content_hash == allowed.content_hash


def test_probe_face_roundoff_coincidence_preserves_geometry_and_stored_identity():
    scene = spec(
        [
            dict(
                kind="rectangle",
                material="PEC",
                x_position=[0.375, 0.625],
                y_position=[0.375, 0.625],
            )
        ]
    )
    scene.sources[0]["x"] = float(np.nextafter(0.375, 1.0))
    original = scene.to_dict()
    original_hash = scene.content_hash
    for n in (128, 256, 512, 1024, 2048):
        sim = scene.build([n, n], reference=True)
        mesh = sim.mesh_uniform()
        assert sim.sources[0][0].x == 0.375
        assert sim.sources[0][0].x in mesh.x
        assert 0.375 in sim.x_anchors and 0.625 in sim.x_anchors
        assert scene.sources[0]["x"] not in sim.x_anchors
    assert scene.to_dict() == original and scene.content_hash == original_hash
    # Truly different probe coordinates must remain distinct mandatory lines.
    scene.sources[0]["x"] = 0.375 + 1e-10
    sim = scene.build([128, 128])
    assert scene.sources[0]["x"] in sim.x_anchors and 0.375 in sim.x_anchors
    with pytest.raises(ValueError, match="Uniform mesh cannot retain"):
        sim.mesh_uniform()


def test_v6_generator_reproducible_resolved_rectangles_wires_and_probes():
    scenes = [make_rich_scene(2026, "train", i) for i in (4, 5, 6)]
    assert scenes[0].content_hash == make_rich_scene(2026, "train", 4).content_hash
    assert any(g["kind"] == "pec_line" for s in scenes for g in s.geometry)
    for s in scenes:
        assert s.pec_policy == "rectangles_and_wires" and s.anchor_probes
        assert all(m["mu_r"] == 1 and m["sigma_h"] == 0 for m in s.materials)
        seen = False
        rectangles = 0
        for g in s.geometry:
            is_pec = g.get("material") == "PEC" or g["kind"] == "pec_line"
            if is_pec:
                seen = True
                assert g["kind"] in ("rectangle", "pec_line")
                for j, axis in enumerate(("x", "y")):
                    values = (
                        g[axis + "_position"]
                        if g["kind"] == "rectangle"
                        else np.atleast_1d(g[axis])
                    )
                    q = np.asarray(values) / s.domain[j] * 64
                    np.testing.assert_allclose(q, np.rint(q), atol=1e-12, rtol=0)
                rectangles += g["kind"] == "rectangle"
            else:
                assert not seen
        assert rectangles >= 1
        sim = s.build([128, 128], reference=True)
        mesh = sim.mesh_uniform()
        assert all(x in mesh.x for x in sim.x_anchors)
        assert all(y in mesh.y for y in sim.y_anchors)
        for probe in [*s.sources, *s.receivers]:
            assert probe["x"] in mesh.x and probe["y"] in mesh.y
    for kwargs in (dict(mu_max=2), dict(magnetic_loss=True), dict(raster_size=256)):
        with pytest.raises(ValueError):
            RichConfig(**kwargs)
