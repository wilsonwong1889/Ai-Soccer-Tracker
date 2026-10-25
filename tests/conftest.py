import shutil
import subprocess

import pytest

from soccer_recorder.sample import make_sample

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)

# Small and short so the suite stays fast; sizes are what the tests assert on.
DURATION = 2.0
FPS = 15


@pytest.fixture(scope="session")
def landscape_clip(tmp_path_factory):
    """H.264 320x180, 2 s at 15 fps, with audio."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    out = tmp_path_factory.mktemp("clips") / "landscape.mp4"
    return make_sample(out, duration=DURATION, width=320, height=180, fps=FPS, codec="h264")


@pytest.fixture(scope="session")
def portrait_clip(tmp_path_factory):
    """HEVC .mov stored 320x180 with 90 degrees display rotation, like an iPhone held upright."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    out = tmp_path_factory.mktemp("clips") / "IMG_0001.MOV"
    try:
        return make_sample(
            out, duration=DURATION, width=320, height=180, fps=FPS, codec="hevc", rotation=90
        )
    except subprocess.CalledProcessError as error:
        if b"display_rotation" in (error.stderr or b""):
            pytest.skip("writing rotation metadata needs FFmpeg 6.0 or newer")
        raise
