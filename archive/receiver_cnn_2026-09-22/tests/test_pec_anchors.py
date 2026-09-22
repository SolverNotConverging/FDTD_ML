import pytest

from fdtdmesh import FDTD_2D_Ez


def simulation():
    return FDTD_2D_Ez(1, 1, 32, 32, 1e9, Nt=1)


def test_face_anchors_exact_idempotent_and_enforced_after_mesh():
    s = simulation()
    s.add_rectangle("PEC", (0.21, 0.49), (0.31, 0.62))
    s.mesh_uniform()
    report = s.add_pec_anchors()
    assert report["added"] == {"x": [0.21, 0.49], "y": [0.31, 0.62]}
    assert s.add_pec_anchors()["added"] == {"x": [], "y": []}
    with pytest.raises(ValueError, match="mandatory"):
        s._prepare()


def test_thin_lines_keep_existing_exact_anchors():
    s = simulation()
    s.add_pec_line(x=0.3, y=(0.2, 0.8))
    before = (s.x_anchors.copy(), s.y_anchors.copy())
    s.add_pec_anchors()
    assert (s.x_anchors, s.y_anchors) == before


def test_features_do_not_claim_axis_alignment_of_curves_or_slants():
    s = simulation()
    s.add_circle("PEC", (0.5, 0.5), 0.1)
    vertices = [(0.1, 0.2), (0.25, 0.35), (0.37, 0.18)]
    s.add_triangle("PEC", vertices)
    assert s.add_pec_anchors()["added"] == {"x": [], "y": []}
    s.add_pec_anchors(mode="features")
    assert {0.4, 0.5, 0.6, 0.1, 0.25, 0.37} <= s.x_anchors
    assert {0.4, 0.5, 0.6, 0.2, 0.35, 0.18} <= s.y_anchors


def test_ordinary_cutout_after_pec_adds_exposed_faces():
    s = simulation()
    s.add_rectangle("PEC", (0.2, 0.8), (0.2, 0.8))
    s.add_rectangle("vacuum", (0.33, 0.52), (0.4, 0.6))
    report = s.add_pec_anchors()
    assert {0.33, 0.52} <= s.x_anchors
    assert {0.4, 0.6} <= s.y_anchors
    assert report["primitives"][-1]["ordinary_override"]
    other = simulation()
    other.add_rectangle("vacuum", (0.33, 0.52), (0.4, 0.6))
    other.add_rectangle("PEC", (0.2, 0.8), (0.2, 0.8))
    other.add_pec_anchors()
    assert 0.33 not in other.x_anchors


def test_close_faces_not_merged_and_invalid_mode_rejected():
    s = simulation()
    s.add_rectangle("PEC", (0.3, 0.30000001), (0.2, 0.8))
    s.add_pec_anchors()
    assert 0.3 in s.x_anchors and 0.30000001 in s.x_anchors
    with pytest.raises(ValueError, match="mode"):
        s.add_pec_anchors(mode="snap")
