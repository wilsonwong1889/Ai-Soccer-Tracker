"""Decode video frames into numpy arrays for detection.

Frames come out small (the long side capped by `max_side`) and at a constant
rate, while the original file stays untouched for full-resolution exports.
ffmpeg applies the file's rotation, so frames are always upright.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import MediaError, VideoInfo, fit_within, probe, require_tool

# Detection input size. YOLO-style models run at 640, but players across a
# full pitch are small, so keep more pixels until R2 benchmarks say otherwise.
DEFAULT_MAX_SIDE = 1280


@dataclass(frozen=True)
class Frame:
    index: int
    # Seconds from the start of the source video.
    time_s: float
    # Height x width x 3, uint8, BGR channel order (OpenCV's convention).
    image: np.ndarray


def iter_frames(
    path: str | Path,
    *,
    max_side: int | None = DEFAULT_MAX_SIDE,
    fps: float | str | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    info: VideoInfo | None = None,
) -> Iterator[Frame]:
    """Yield upright BGR frames from `path`.

    `fps` resamples to a constant rate (default: the source's average rate),
    which also evens out iPhone variable-frame-rate recordings so tracking
    sees a steady time step. `start_s`/`duration_s` decode just a window.
    """
    path = Path(path)
    info = info or probe(path)
    width, height = fit_within(info.display_width, info.display_height, max_side)
    rate = Fraction(str(fps)) if fps is not None else Fraction(info.avg_frame_rate)
    if rate <= 0:
        raise ValueError("fps must be positive")

    command = [require_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.6f}"]
    command += ["-i", str(path)]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.6f}"]
    command += [
        "-map", "0:v:0",
        "-vf", f"fps={rate},scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:
            index = 0
            while True:
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                # bytearray keeps the array writable, so later stages can draw on it.
                image = np.frombuffer(bytearray(data), dtype=np.uint8).reshape(height, width, 3)
                yield Frame(index=index, time_s=start_s + float(index / rate), image=image)
                index += 1
            returncode = process.wait()
            if returncode != 0:
                stderr.seek(0)
                detail = stderr.read().decode(errors="replace").strip()
                raise MediaError(f"ffmpeg failed decoding {path.name}: {detail or returncode}")
        finally:
            # Runs when the caller stops early too: don't leave ffmpeg running.
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
