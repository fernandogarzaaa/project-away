import numpy as np
import pytest

from away.hands import landmarks as L
from away.hands.synthetic import make_hand, pose


def test_as_array_pads_2d_and_validates():
    a = L.as_array(np.zeros((21, 2)))
    assert a.shape == (21, 3)
    for bad in (np.zeros((20, 3)), np.zeros((21, 4)), np.zeros(21)):
        with pytest.raises(ValueError):
            L.as_array(bad)


def test_normalize_translation_scale_invariant():
    a = pose("point", center=(0.3, 0.4), scale=0.1)
    b = pose("point", center=(0.7, 0.2), scale=0.2)
    na, nb = L.normalize(a), L.normalize(b)
    assert np.allclose(na[:, :2], nb[:, :2], atol=1e-9)
    assert np.allclose(na[L.WRIST], 0)
    assert L.palm_size(na) == pytest.approx(1.0)


def test_normalize_mirrors_left_hand():
    r = pose("two", handedness="Right")
    left = pose("two", handedness="Left")
    assert np.allclose(L.normalize(r)[:, :2], L.normalize(left, "Left")[:, :2], atol=1e-9)
    assert not np.allclose(L.normalize(left, "Right")[:, 0], L.normalize(r)[:, 0])


def test_normalize_degenerate():
    with pytest.raises(ValueError):
        L.normalize(np.zeros((21, 3)))


def test_palm_size():
    lm = np.zeros((21, 3))
    lm[L.MIDDLE_MCP] = (3, 4, 9)
    assert L.palm_size(lm) == 5.0


def test_pinch_ratio_open_vs_closed():
    assert L.pinch_ratio(pose("pinch")) < 0.05
    assert L.pinch_ratio(pose("pinch_open")) > 0.5
    lm = np.zeros((21, 3))
    lm[L.THUMB_TIP] = (1, 0, 0)
    lm[L.MIDDLE_MCP] = (0, 2, 0)
    assert L.pinch_ratio(lm) == pytest.approx(0.5)
    assert L.pinch_ratio(np.zeros((21, 3))) == float("inf")


@pytest.mark.parametrize("name,expected", [("point", "point"), ("palm", "palm"), ("fist", "fist"), ("two", "two")])
@pytest.mark.parametrize("hand", ["Right", "Left"])
def test_classify_pose(name, expected, hand):
    assert L.classify_pose(pose(name, handedness=hand, rotation=0.2)) == expected


def test_classify_other():
    lm = make_hand(curls={"index": 1.0, "middle": 0.0, "ring": 1.0, "pinky": 0.0})
    assert L.classify_pose(lm) == "other"
    # index + pinky ("rock") is not "point"
    lm = make_hand(curls={"index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0})
    assert L.classify_pose(lm) == "other"
    # three fingers is not "two"
    lm = make_hand(curls={"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0})
    assert L.classify_pose(lm) == "other"


def test_finger_extended_threshold():
    lm = np.zeros((21, 3))
    lm[L.INDEX_PIP] = (0, 1.0, 0)
    lm[L.INDEX_TIP] = (0, 1.1, 0)  # exactly 1.1x is NOT extended
    assert not L.finger_extended(lm, "index")
    lm[L.INDEX_TIP] = (0, 1.11, 0)
    assert L.finger_extended(lm, "index")


def test_infer_side():
    assert L.infer_side(pose("palm")) == "Right"
    assert L.infer_side(pose("palm", handedness="Left")) == "Left"
    lm = np.zeros((21, 3))
    assert L.infer_side(lm) == "Right"  # tie -> Right


def test_static_and_sequence_features():
    lm = pose("point")
    f = L.static_features(lm)
    assert f.shape == (42,)
    seq = np.stack([pose("palm", center=(0.3 + 0.01 * i, 0.5), scale=0.1) for i in range(5)])
    feats = L.sequence_features(seq)
    assert feats.shape == (5, 44)
    assert np.allclose(feats[0, -2:], 0)
    assert feats[1, -2] == pytest.approx(0.01 / 0.1, rel=1e-6)  # palm units per frame
    assert feats[1, -1] == pytest.approx(0.0, abs=1e-9)
    left = L.sequence_features(seq, "Left")
    assert left[1, -2] == pytest.approx(-0.1, rel=1e-6)
    with pytest.raises(ValueError):
        L.normalize_sequence(np.zeros((21, 3)))
