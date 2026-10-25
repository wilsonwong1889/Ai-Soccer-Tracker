import numpy as np
import pytest
from conftest import DURATION, FPS, needs_ffmpeg

from soccer_recorder.frames import iter_frames

pytestmark = needs_ffmpeg

PITCH_BGR = (0x32, 0x7D, 0x2E)  # the synthetic pitch colour 0x2e7d32, as B, G, R


def test_frames_are_bgr_at_source_rate(landscape_clip):
    frames = list(iter_frames(landscape_clip, max_side=None))
    assert len(frames) == pytest.approx(DURATION * FPS, abs=1)
    first = frames[0]
    assert first.image.shape == (180, 320, 3)
    assert first.image.dtype == np.uint8
    assert first.image.flags.writeable
    # A point on the pitch away from lines and movers.
    assert np.allclose(first.image[150, 300], PITCH_BGR, atol=12)
    assert [f.index for f in frames] == list(range(len(frames)))
    assert frames[3].time_s == pytest.approx(3 / FPS)


def test_frames_downscale_and_resample(landscape_clip):
    frames = list(iter_frames(landscape_clip, max_side=160, fps=5))
    assert frames[0].image.shape == (90, 160, 3)
    assert len(frames) == pytest.approx(DURATION * 5, abs=1)
    assert frames[1].time_s == pytest.approx(0.2)


def test_frames_window(landscape_clip):
    frames = list(iter_frames(landscape_clip, start_s=1.0, duration_s=0.5))
    assert len(frames) == pytest.approx(0.5 * FPS, abs=1)
    assert frames[0].time_s == pytest.approx(1.0)


def test_frames_are_upright_for_rotated_video(portrait_clip):
    frame = next(iter_frames(portrait_clip, max_side=None))
    assert frame.image.shape == (320, 180, 3)


def test_stopping_early_does_not_hang_or_raise(landscape_clip):
    frames = iter_frames(landscape_clip)
    next(frames)
    frames.close()
