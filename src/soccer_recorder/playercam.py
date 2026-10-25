"""Player cam: a virtual camera that follows the player (R6).

The phone films the whole pitch. The player cam crops it around the
followed player, as if a camera operator were following them:

- it zooms so the player fills about a third of the picture;
- it pans and zooms smoothly. The whole path is known in advance, so the
  smoothing looks both ways in time and never lags behind;
- while the player is hidden or lost, it holds position and eases out to
  the whole pitch, then eases back in when they reappear;
- it never zooms in beyond a quarter of the frame height, to keep the
  picture sharp.

Frames are cut from the original video (up to `source_max_side`), so the
crop keeps detail the detection frames don't have. The game's sound is
kept. Output is landscape (1280x720) or vertical for phones (720x1280).
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detect import Progress
from .frames import iter_frames
from .media import VideoInfo, probe, require_tool
from .overlay import HIGHLIGHT_RGB, MAX_RENDER_FPS, draw_ring, label_sprite, paste, place_label
from .render import VideoWriter

LANDSCAPE = (1280, 720)
VERTICAL = (720, 1280)
DEFAULT_SOURCE_MAX_SIDE = 2560


@dataclass(frozen=True)
class CamParams:
    out_size: tuple[int, int] = LANDSCAPE
    player_share: float = 0.33  # player height as a share of the picture height
    min_crop_share: float = 0.25  # never tighter than this share of the frame height
    smooth_s: float = 0.6  # position smoothing
    zoom_smooth_s: float = 1.2  # zoom smoothing
    lost_wide_s: float = 1.5  # time to ease out to the whole pitch when the player is lost
    source_max_side: int | None = DEFAULT_SOURCE_MAX_SIDE


DEFAULT_CAM = CamParams()


def _gauss(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 3:
        return values
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    num = np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")
    return num / kernel.sum()


def camera_path(trajectory, times: np.ndarray, frame_size: tuple[int, int],
                params: CamParams = DEFAULT_CAM) -> np.ndarray:
    """Crop boxes (x1, y1, x2, y2) in source pixels, one per output time."""
    width, height = frame_size
    aspect = params.out_size[0] / params.out_size[1]
    n = len(times)
    fps = 1.0 / float(np.median(np.diff(times))) if n > 1 else 30.0
    cx, cy, crop_h = np.zeros(n), np.zeros(n), np.zeros(n)
    full_h = min(height, width / aspect)  # the tallest crop with the right shape
    known = [trajectory.box_at(float(t)) for t in times]
    first = next((k for k in known if k is not None), None)
    last_x, last_y = ((first[0][0] + first[0][2]) / 2, (first[0][1] + first[0][3]) / 2) \
        if first else (width / 2, height / 2)
    lost_for = 0.0
    for i, found in enumerate(known):
        if found is not None:
            x1, y1, x2, y2 = found[0]
            last_x, last_y = (x1 + x2) / 2, (y1 + y2) / 2
            wanted = (y2 - y1) / params.player_share
            lost_for = 0.0
        else:
            lost_for += 1.0 / fps
            # Ease out towards the whole pitch while the player can't be seen.
            ease = min(1.0, lost_for / params.lost_wide_s)
            wanted = crop_h[i - 1] + (full_h - crop_h[i - 1]) * ease if i else full_h
            if first is None:
                last_x, last_y = width / 2, height / 2
        cx[i], cy[i] = last_x, last_y
        crop_h[i] = min(full_h, max(params.min_crop_share * height, wanted))
    crop_h = _gauss(crop_h, params.zoom_smooth_s * fps)
    crop_h = np.clip(crop_h, params.min_crop_share * height, full_h)
    crop_w = crop_h * aspect
    # Stop at the frame's edges before smoothing, so the camera eases to a stop
    # there instead of halting abruptly.
    cx = np.clip(cx, crop_w / 2, width - crop_w / 2)
    cy = np.clip(cy, crop_h / 2, height - crop_h / 2)
    cx, cy = _gauss(cx, params.smooth_s * fps), _gauss(cy, params.smooth_s * fps)
    x1 = np.clip(cx - crop_w / 2, 0, width - crop_w)
    y1 = np.clip(cy - crop_h / 2, 0, height - crop_h)
    return np.stack([x1, y1, x1 + crop_w, y1 + crop_h], axis=1)


def framing_report(trajectory, times: np.ndarray, crops: np.ndarray,
                   frame_size: tuple[int, int]) -> dict:
    """How well the camera kept the player in shot, and how smoothly it moved (R6).

    - centred: share of frames where the player is seen and their centre is in
      the middle 60% of the picture, both ways. Frames where the player is so
      near the edge of the whole pitch view that no crop could centre them
      are left out, and counted as at_edge;
    - acceleration: how fast the camera's pan speeds up or slows down, in
      picture widths per second squared (99th percentile and maximum).
    """
    width, height = frame_size
    centred = seen = at_edge = 0
    for i, t in enumerate(times):
        found = trajectory.box_at(float(t))
        if found is None or found[1] != "seen":
            continue
        box = found[0]
        x1, y1, x2, y2 = crops[i]
        px, py = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        # Could a crop this size centre them at all?
        if not (0.3 * (x2 - x1) <= px <= width - 0.3 * (x2 - x1)
                and 0.3 * (y2 - y1) <= py <= height - 0.3 * (y2 - y1)):
            at_edge += 1
            continue
        seen += 1
        centred += 0.2 <= (px - x1) / (x2 - x1) <= 0.8 and 0.2 <= (py - y1) / (y2 - y1) <= 0.8
    report = {"frames_seen": seen + at_edge, "at_edge": at_edge,
              "centred": round(centred / seen, 3) if seen else None}
    if len(times) > 2:
        dt = float(np.median(np.diff(times)))
        widths = crops[:, 2] - crops[:, 0]
        centre = (crops[:, 0] + crops[:, 2]) / 2
        speed = np.diff(centre) / widths[:-1] / dt  # pan, in picture widths per second
        accel = np.abs(np.diff(speed)) / dt
        report["acceleration_p99"] = round(float(np.percentile(accel, 99)), 2)
        report["acceleration_max"] = round(float(accel.max()), 2)
    return report


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        shrink = image.shape[1] > size[0]
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    except ImportError:  # Pillow, also in the detect extra
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


def add_audio(video_only: Path, source: Path, start_s: float, duration_s: float,
              out_path: Path, has_audio: bool) -> None:
    """Copy the source's sound for [start, start + duration] onto the rendered video."""
    if not has_audio:
        Path(video_only).replace(out_path)
        return
    result = subprocess.run(
        [require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-i", str(video_only),
         "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(source),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-shortest", "-movflags", "+faststart", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"adding the sound failed: {result.stderr.strip()}")


def render_playercam(
    video: str | Path,
    trajectory,
    target,
    out_path: str | Path,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    params: CamParams = DEFAULT_CAM,
    caption: str | None = None,
    info: VideoInfo | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> int:
    """Render the player cam for [start_s, end_s] (default: the analysed stretch)."""
    video, out_path = Path(video), Path(out_path)
    info = info or probe(video)
    times = trajectory.times
    start = float(times[0]) if start_s is None else max(0.0, start_s)
    end = (float(times[-1]) + 1.0 / max(info.fps, 1.0)) if end_s is None else min(
        end_s, info.duration_s)
    if end <= start:
        raise ValueError("nothing to render")
    rate = min(info.fps, MAX_RENDER_FPS)
    out_w, out_h = params.out_size
    sprite = label_sprite(target.label, max(16, round(out_h / 24)))
    caption_sprite = (label_sprite(caption, max(16, round(out_h / 26)), (255, 255, 255),
                                   pointer=False) if caption else None)
    ring_bgr = HIGHLIGHT_RGB[::-1]
    written = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        silent = Path(temp) / "video.mp4"
        frames = iter_frames(video, max_side=params.source_max_side, fps=rate, start_s=start,
                             duration_s=end - start, info=info)
        writer = None
        try:
            path = None
            for frame in frames:
                image = frame.image
                height, width = image.shape[:2]
                if path is None:
                    out_times = start + np.arange(int(np.ceil((end - start) * rate)) + 2) / rate
                    path = camera_path(trajectory, out_times, (info.display_width,
                                                               info.display_height), params)
                    to_frame = width / info.display_width
                    writer = VideoWriter(silent, out_w, out_h, rate)
                x1, y1, x2, y2 = path[min(frame.index, len(path) - 1)] * to_frame
                crop = image[int(y1):max(int(y1) + 2, int(y2)), int(x1):max(int(x1) + 2, int(x2))]
                picture = np.ascontiguousarray(_resize(crop, (out_w, out_h)))
                found = trajectory.box_at(frame.time_s)
                if found is not None:
                    box, state = found
                    zoom = out_w / max(x2 - x1, 1.0)
                    mapped = ((box[0] * to_frame - x1) * zoom, (box[1] * to_frame - y1) * zoom,
                              (box[2] * to_frame - x1) * zoom, (box[3] * to_frame - y1) * zoom)
                    seen = state == "seen"
                    draw_ring(picture, mapped, ring_bgr, 3.5 if seen else 2.0)
                    place_label(picture, sprite, mapped, 1.0 if seen else 0.65)
                if caption_sprite is not None:
                    paste(picture, caption_sprite, round(out_w * 0.03), round(out_h * 0.04))
                writer.write(picture)
                written += 1
                if on_progress is not None:
                    on_progress(Progress(frames=written, video_s=frame.time_s - start,
                                         total_video_s=end - start,
                                         elapsed_s=time.perf_counter() - started))
            if writer is not None:
                writer.close()
                writer = None
        finally:
            frames.close()
            if writer is not None:
                writer.abort()
        if not written:
            raise ValueError("no frames in that stretch")
        add_audio(silent, video, start, end - start, out_path, info.has_audio)
    return written
