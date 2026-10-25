"""Write frames to video and draw on them, with numpy and FFmpeg only."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import MediaError, require_tool

# BGR colours that stand out on a green pitch or a gym floor.
LABEL_COLOURS = {
    "person": (0, 200, 255),  # amber
    "sports ball": (255, 0, 255),  # magenta
}
DEFAULT_COLOUR = (255, 255, 255)


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a rectangle outline on a BGR image in place, clipped to the image."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, x2)), min(width - 1, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height - 1, max(y1, y2))
    if x1 > x2 or y1 > y2:
        return
    t = max(1, thickness)
    image[y1 : min(y1 + t, y2 + 1), x1 : x2 + 1] = colour
    image[max(y2 - t + 1, y1) : y2 + 1, x1 : x2 + 1] = colour
    image[y1 : y2 + 1, x1 : min(x1 + t, x2 + 1)] = colour
    image[y1 : y2 + 1, max(x2 - t + 1, x1) : x2 + 1] = colour


def draw_boxes(image: np.ndarray, boxes: Iterable, scale: float = 1.0) -> None:
    """Draw detections (anything with x1, y1, x2, y2, score, label) on `image`.

    `scale` maps box coordinates to image pixels. Confident boxes are drawn
    thicker than doubtful ones.
    """
    for box in boxes:
        colour = LABEL_COLOURS.get(box.label, DEFAULT_COLOUR)
        thickness = 2 if box.score >= 0.5 else 1
        draw_box(
            image,
            (box.x1 * scale, box.y1 * scale, box.x2 * scale, box.y2 * scale),
            colour,
            thickness,
        )


class VideoWriter:
    """Pipe BGR frames into an H.264 MP4 through FFmpeg.

        with VideoWriter(path, width, height, fps) as writer:
            writer.write(frame)
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float | str | Fraction):
        if width % 2 or height % 2:
            raise ValueError("width and height must be even for H.264")
        self.path = Path(path)
        self.width, self.height = width, height
        self.frames = 0
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [
                require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=self._stderr,
        )

    def write(self, image: np.ndarray) -> None:
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"frame is {image.shape} {image.dtype}, writer expects "
                f"({self.height}, {self.width}, 3) uint8"
            )
        self._process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()
        returncode = self._process.wait()
        self._stderr.seek(0)
        detail = self._stderr.read().decode(errors="replace").strip()
        self._stderr.close()
        if returncode != 0:
            raise MediaError(f"ffmpeg failed writing {self.path.name}: {detail or returncode}")

    def abort(self) -> None:
        """Stop FFmpeg and delete the partial file (used when something failed)."""
        self._process.kill()
        self._process.wait()
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        self._stderr.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()  # don't mask the original error
        else:
            self.close()
