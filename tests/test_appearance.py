import numpy as np
import pytest

from soccer_recorder.appearance import FEATURE_SIZE, bgr_to_hsv, describe, mean_feature, similarity


def test_hue_of_primary_colours():
    pixels = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0], [128, 128, 128]]], dtype=np.uint8)
    hue, sat, val = bgr_to_hsv(pixels)
    assert hue[0, :3] == pytest.approx([0.0, 1 / 3, 2 / 3], abs=1e-3)  # red, green, blue
    assert sat[0, 3] == 0.0 and val[0, 3] == pytest.approx(128 / 255)


def player(shirt, shorts=(240, 240, 240), hair=(30, 30, 30), size=(60, 22)):
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: int(0.18 * height)] = hair
    image[int(0.18 * height): int(0.55 * height)] = shirt
    image[int(0.55 * height):] = shorts
    return image


def test_same_kit_is_similar_and_other_team_is_not():
    red, blue = (40, 40, 220), (200, 80, 30)
    a = describe(player(red), (0, 0, 22, 60))
    b = describe(player(red), (0, 0, 22, 60))
    # Teammates wear the same kit: only hair (and skin) can differ.
    teammate = describe(player(red, hair=(90, 200, 230)), (0, 0, 22, 60))
    # The other team's kit differs in shirt and shorts.
    opponent = describe(player(blue, shorts=(20, 20, 20)), (0, 0, 22, 60))
    assert a.shape == (FEATURE_SIZE,)
    assert similarity(a, b) == pytest.approx(1.0, abs=1e-3)
    assert similarity(a, opponent) < similarity(a, teammate) < similarity(a, b)
    assert similarity(a, opponent) < 0.6


def test_tiny_or_missing_players_have_no_description():
    image = player((40, 40, 220))
    assert describe(image, (0, 0, 2, 4)) is None
    assert similarity(None, describe(image, (0, 0, 22, 60))) == 0.0
    assert mean_feature([None, None]) is None
