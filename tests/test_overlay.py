from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL", reason="detect extra not installed")

from PIL import ImageFont  # noqa: E402

from soccer_recorder.overlay import (  # noqa: E402
    FONT_CANDIDATES,
    covers,
    draw_ring,
    label_sprite,
    paste,
    pick_font,
    place_label,
)


def test_builtin_font_lacks_accents_so_a_fuller_font_is_preferred():
    builtin = ImageFont.load_default(size=20)
    assert covers(builtin, "Emma #10")
    assert not covers(builtin, "Zoë")  # why FONT_CANDIDATES exists
    assert pick_font("Zoë #7", 20, candidates=()).getbbox("Z")  # falls back, still draws


@pytest.mark.skipif(not any(Path(p).is_file() for p in FONT_CANDIDATES),
                    reason="no system font from the list on this machine")
def test_picked_font_covers_accented_names():
    font = pick_font("Siobhán Ó'Brien  #4", 20)
    assert covers(font, "Siobhán Ó'Brien  #4")


def test_label_sprite_is_an_opaque_tag_with_transparent_corners():
    sprite = label_sprite("Emma  #10", 20)
    height, width, channels = sprite.shape
    assert channels == 4 and width > height
    assert sprite[0, 0, 3] == 0  # rounded corner: transparent
    assert sprite[height // 3, width // 2, 3] > 200  # body: opaque


def test_paste_clips_at_every_edge():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    sprite = np.zeros((10, 20, 4), dtype=np.uint8)
    sprite[...] = (0, 0, 255, 255)
    for x, y in [(-10, -5), (50, 35), (-100, 0), (0, 100)]:
        paste(image, sprite, x, y)  # never raises
    assert image[0, 0].tolist() == [0, 0, 255]
    assert image[39, 59].tolist() == [0, 0, 255]


def test_place_label_stays_inside_the_frame():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    sprite = np.full((12, 40, 4), 255, dtype=np.uint8)
    place_label(image, sprite, (85, 5, 99, 60))  # player at the top-right corner
    painted = np.argwhere(image.any(axis=2))
    assert painted[:, 1].max() <= 99 and painted[:, 0].min() >= 0
    assert painted[:, 1].min() == 60  # pushed left to fit


def test_draw_ring_marks_the_feet_only():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_ring(image, (40, 20, 60, 80), (0, 200, 255))
    painted = np.argwhere(image.any(axis=2))
    assert painted[:, 0].min() >= 70  # around y = 80, not up the body
    assert image[80, 50].tolist() == [0, 0, 0]  # a ring, not a filled disc
