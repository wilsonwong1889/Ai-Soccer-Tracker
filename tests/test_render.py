import numpy as np
import pytest
from conftest import needs_ffmpeg

from soccer_recorder.media import probe
from soccer_recorder.render import VideoWriter, draw_box


def test_draw_box_outlines_and_leaves_inside_untouched():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    draw_box(image, (2, 3, 12, 15), (0, 0, 255), thickness=1)
    assert (image[3, 2:13] == (0, 0, 255)).all()  # top edge
    assert (image[15, 2:13] == (0, 0, 255)).all()  # bottom edge
    assert (image[3:16, 2] == (0, 0, 255)).all()  # left edge
    assert (image[3:16, 12] == (0, 0, 255)).all()  # right edge
    assert (image[5:14, 4:11] == 0).all()  # inside
    assert (image[0, :] == 0).all()  # outside


def test_draw_box_clips_to_the_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_box(image, (-5, -5, 50, 50), (255, 255, 255))
    assert image[0, 0].tolist() == [255, 255, 255]
    assert image[9, 9].tolist() == [255, 255, 255]
    draw_box(image, (20, 20, 30, 30), (1, 2, 3))  # entirely outside: no error
    assert not (image == (1, 2, 3)).all(axis=2).any()


@needs_ffmpeg
def test_video_writer_writes_playable_mp4(tmp_path):
    path = tmp_path / "out.mp4"
    with VideoWriter(path, 64, 36, 10) as writer:
        for value in range(20):
            writer.write(np.full((36, 64, 3), value * 10, dtype=np.uint8))
    info = probe(path)
    assert (info.display_width, info.display_height) == (64, 36)
    assert info.fps == pytest.approx(10)
    assert info.duration_s == pytest.approx(2.0, abs=0.15)
    assert writer.frames == 20


@needs_ffmpeg
def test_video_writer_rejects_wrong_frame_and_cleans_up(tmp_path):
    path = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="expects"):
        with VideoWriter(path, 64, 36, 10) as writer:
            writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    assert not path.exists()


def test_video_writer_needs_even_size(tmp_path):
    with pytest.raises(ValueError, match="even"):
        VideoWriter(tmp_path / "x.mp4", 63, 36, 10)
