"""Decode video frames into numpy arrays for detection.

Frames come out small (the long side capped by `max_side`) and at a constant
rate, while the original file stays untouched for full-resolution exports.
ffmpeg applies the file's rotation, so frames are always upright.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .media import MediaError, VideoInfo, fit_within, probe, require_tool, run_ffprobe

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


def read_image(path: str | Path, max_side: int | None = None) -> tuple[np.ndarray, int, int]:
    """Load an image as BGR, scaled like detection frames.

    Returns (image, original_width, original_height), so boxes found on the
    scaled image can be mapped back to the original's pixels.
    """
    path = Path(path)
    data = run_ffprobe(path)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None or not stream.get("width") or not stream.get("height"):
        raise MediaError(f"{path.name} is not an image")
    original_w, original_h = int(stream["width"]), int(stream["height"])
    width, height = fit_within(original_w, original_h, max_side)
    result = subprocess.run(
        [
            require_tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or len(result.stdout) != width * height * 3:
        detail = result.stderr.decode(errors="replace").strip()
        raise MediaError(f"ffmpeg could not decode {path.name}: {detail or 'short read'}")
    image = np.frombuffer(bytearray(result.stdout), dtype=np.uint8).reshape(height, width, 3)
    return image, original_w, original_h


def sample_frames(
    path: str | Path,
    out_dir: str | Path,
    *,
    every_s: float,
    prefix: str,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> list[dict]:
    """Save one full-resolution JPEG every `every_s` seconds, for labelling.

    Files are named `<prefix>_<n>.jpg`; use the asset id as the prefix so
    frames from different games never collide in one labelling task. Also
    writes frames.json listing each file with its time in the video.
    """
    if every_s <= 0:
        raise ValueError("every_s must be positive")
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{prefix}_*.jpg"):  # from an earlier, longer sampling
        stale.unlink()
    command = [require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.6f}"]
    command += ["-i", str(path)]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.6f}"]
    command += [
        "-map", "0:v:0", "-vf", f"fps=1/{every_s}", "-q:v", "2",
        "-start_number", "0", str(out_dir / f"{prefix}_%05d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(f"ffmpeg could not sample {path.name}: {result.stderr.strip()}")
    files = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    frames = [
        {"file": file.name, "t": round(start_s + n * every_s, 3)}
        for n, file in enumerate(files)
    ]
    (out_dir / "frames.json").write_text(json.dumps(
        {"source": path.name, "every_s": every_s, "frames": frames}, indent=2
    ) + "\n")
    return frames
