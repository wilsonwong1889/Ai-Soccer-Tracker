"""Read video metadata with ffprobe.

FFmpeg is the one media dependency: it decodes the HEVC/H.264 files iPhones
record on every platform we care about, and the same binary will render
exports in R6. Everything here shells out to `ffprobe`/`ffmpeg` on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

# Transfer characteristics that mean the clip is HDR (iPhones record HLG by
# default). Detection models are trained on SDR images, so these frames look
# washed out unless they are tone-mapped first.
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}

# avg and nominal frame rates further apart than this mean variable frame rate.
VFR_TOLERANCE = 0.01


class MediaError(Exception):
    """A file could not be read as a usable video."""


class ToolMissingError(MediaError):
    """ffmpeg or ffprobe is not installed."""


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolMissingError(
            f"{name} not found on PATH. Install FFmpeg (macOS: `brew install ffmpeg`)."
        )
    return path


@dataclass(frozen=True)
class VideoInfo:
    """What we know about a source video before any analysis."""

    filename: str
    size_bytes: int
    container: str
    duration_s: float
    codec: str
    pix_fmt: str | None
    # Coded size, as stored in the file.
    width: int
    height: int
    # Clockwise degrees the player rotates the picture for display (0/90/180/270).
    rotation: int
    # Upright size, after rotation. Frames decoded by ffmpeg come out this size.
    display_width: int
    display_height: int
    avg_frame_rate: str
    fps: float
    is_vfr: bool
    frame_count: int | None
    bit_rate: int | None
    color_transfer: str | None
    is_hdr: bool
    has_audio: bool
    creation_time: str | None
    device: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VideoInfo:
        return cls(**data)


def _fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _int(text) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _float(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict) -> int:
    """Clockwise display rotation, normalised to 0/90/180/270.

    ffprobe reports the display matrix as a counter-clockwise `rotation`
    (FFmpeg 5+); older builds put a clockwise `rotate` tag on the stream.
    """
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return round(-float(side_data["rotation"])) % 360
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is not None:
        return round(float(rotate)) % 360
    return 0


def _device(tags: dict) -> str | None:
    make = tags.get("com.apple.quicktime.make")
    model = tags.get("com.apple.quicktime.model")
    parts = [part for part in (make, model) if part]
    return " ".join(parts) or None


def run_ffprobe(path: Path) -> dict:
    ffprobe = require_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown error"
        raise MediaError(f"ffprobe could not read {path.name}: {detail}")
    return json.loads(result.stdout)


def probe(path: str | Path) -> VideoInfo:
    """Describe the first video stream of `path`, or raise MediaError."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"No such file: {path}")

    data = run_ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise MediaError(f"{path.name} has no video stream")

    width, height = _int(video.get("width")), _int(video.get("height"))
    if not width or not height:
        raise MediaError(f"{path.name} has no frame size")

    duration = _float(fmt.get("duration")) or _float(video.get("duration")) or 0.0
    if duration <= 0:
        raise MediaError(f"{path.name} has no duration (empty or truncated file?)")

    avg = _fraction(video.get("avg_frame_rate"))
    nominal = _fraction(video.get("r_frame_rate"))
    rate = avg or nominal
    if rate is None:
        raise MediaError(f"{path.name} has no frame rate")
    is_vfr = bool(
        avg and nominal and abs(float(avg) - float(nominal)) / float(nominal) > VFR_TOLERANCE
    )

    rotation = _rotation(video)
    sideways = rotation in (90, 270)
    transfer = video.get("color_transfer")
    is_hdr = transfer in HDR_TRANSFERS
    tags = {**fmt.get("tags", {}), **video.get("tags", {})}

    warnings = []
    if is_hdr:
        warnings.append(
            f"HDR video ({transfer}): detection frames will look washed out until "
            "tone-mapping is added. For test recordings, turn off "
            "Settings > Camera > Record Video > HDR Video."
        )
    if is_vfr:
        warnings.append(
            f"Variable frame rate (average {float(avg):.3f} fps, nominal {float(nominal):.3f}): "
            "frames are resampled to a constant rate for analysis."
        )
    if sideways:
        warnings.append(
            f"Recorded in portrait (rotated {rotation} degrees): a sideline tripod "
            "recording is normally landscape."
        )

    return VideoInfo(
        filename=path.name,
        size_bytes=path.stat().st_size,
        container=fmt.get("format_name", "unknown"),
        duration_s=duration,
        codec=video.get("codec_name", "unknown"),
        pix_fmt=video.get("pix_fmt"),
        width=width,
        height=height,
        rotation=rotation,
        display_width=height if sideways else width,
        display_height=width if sideways else height,
        avg_frame_rate=str(rate),
        fps=float(rate),
        is_vfr=is_vfr,
        frame_count=_int(video.get("nb_frames")),
        bit_rate=_int(fmt.get("bit_rate")),
        color_transfer=transfer,
        is_hdr=is_hdr,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        creation_time=tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"),
        device=_device(tags),
        warnings=warnings,
    )


def fit_within(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """Scale (width, height) down so the long side is at most `max_side`.

    Never upscales. Both sides are rounded down to even numbers, which
    H.264/yuv420p requires.
    """
    if max_side is not None and max_side < 2:
        raise ValueError("max_side must be at least 2")
    scale = 1.0
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)

    def even(value: float) -> int:
        # The epsilon keeps 1279.9999 (float error) from flooring to 1278.
        return max(2, int(value + 1e-6) // 2 * 2)

    return even(width * scale), even(height * scale)
