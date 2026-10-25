"""Draw the followed player: a ring at their feet and a "Name #10" tag above.

The tag is drawn once with Pillow (in the detect extra), then stamped onto
each frame with numpy, so rendering a whole game stays quick. Frames are
rendered at the video's own rate (up to 30 fps), with the player's position
interpolated between analysed frames, so the tag moves smoothly.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .detect import Progress
from .frames import DEFAULT_MAX_SIDE, iter_frames
from .media import VideoInfo, probe
from .render import VideoWriter, draw_box

HIGHLIGHT_RGB = (255, 196, 0)  # amber
TEXT_RGB = (20, 20, 20)
MAX_RENDER_FPS = 30.0
ID_COLOURS_BGR = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245),
    (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250),
]


# Fonts already on the computer, best first. Names come with accents (Zoë,
# José, Siobhán) or in other scripts, and Pillow's built-in font only has
# basic Latin letters, so the first font that has every letter wins.
# (Fonts are only used to draw labels locally. A hosted service would ship
# its own openly licensed font, such as Noto Sans.)
FONT_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
)


def _missing_glyph(font) -> bytes:
    return bytes(font.getmask("\U0010FFFD"))  # an unassigned character: the font's "missing" box


def covers(font, text: str) -> bool:
    """Whether `font` has a real glyph for every visible character of `text`."""
    missing = _missing_glyph(font)
    return all(ch.isspace() or bytes(font.getmask(ch)) != missing for ch in text)


@functools.lru_cache(maxsize=32)
def pick_font(text: str, font_px: int, candidates: tuple[str, ...] = FONT_CANDIDATES):
    """The first installed font that can write `text`; Pillow's built-in one otherwise."""
    from PIL import ImageFont

    for path in candidates:
        if not Path(path).is_file():
            continue
        try:
            font = ImageFont.truetype(path, font_px)
        except OSError:
            continue
        if covers(font, text):
            return font
    return ImageFont.load_default(size=font_px)


def label_sprite(text: str, font_px: int, rgb=HIGHLIGHT_RGB) -> np.ndarray:
    """A rounded tag with a pointer underneath, as BGRA uint8."""
    from PIL import Image, ImageDraw

    font = pick_font(text, font_px)
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = round(font_px * 0.55), round(font_px * 0.35)
    pointer = max(4, round(font_px * 0.45))
    width = right - left + 2 * pad_x
    body = bottom - top + 2 * pad_y
    image = Image.new("RGBA", (width, body + pointer), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, width - 1, body - 1], radius=round(body * 0.3),
                           fill=(*rgb, 240))
    mid = width // 2
    draw.polygon([(mid - pointer, body - 1), (mid + pointer, body - 1), (mid, body + pointer - 1)],
                 fill=(*rgb, 240))
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=(*TEXT_RGB, 255))
    rgba = np.array(image)
    return rgba[..., [2, 1, 0, 3]].copy()  # RGBA -> BGRA


def paste(image: np.ndarray, sprite: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite onto a BGR image at (x, y), clipped to the image."""
    height, width = image.shape[:2]
    sh, sw = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + sw), min(height, y + sh)
    if x0 >= x1 or y0 >= y1:
        return
    part = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = part[..., 3:4].astype(np.float32) / 255.0 * opacity
    region = image[y0:y1, x0:x1].astype(np.float32)
    image[y0:y1, x0:x1] = (part[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)


def draw_ring(image: np.ndarray, box, bgr, thickness: float = 2.5) -> None:
    """An ellipse on the ground around the player's feet."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, y2
    a = max(6.0, 0.75 * (x2 - x1))
    b = max(3.0, 0.28 * a)
    height, width = image.shape[:2]
    left, right = int(max(0, cx - a - thickness)), int(min(width, cx + a + thickness + 1))
    top, bottom = int(max(0, cy - b - thickness)), int(min(height, cy + b + thickness + 1))
    if left >= right or top >= bottom:
        return
    ys, xs = np.mgrid[top:bottom, left:right]
    r = np.sqrt(((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2)
    ring = np.abs(r - 1.0) * min(a, b) <= thickness / 2
    image[top:bottom, left:right][ring] = bgr


def place_label(image: np.ndarray, sprite: np.ndarray, box, opacity: float = 1.0) -> None:
    """Put the tag centred above the player's head, kept inside the frame."""
    x1, y1, x2, _ = box
    height, width = image.shape[:2]
    sh, sw = sprite.shape[:2]
    gap = max(2, round(sh * 0.15))
    x = int(round((x1 + x2) / 2 - sw / 2))
    y = int(round(y1 - sh - gap))
    x = min(max(x, 0), max(0, width - sw))
    y = max(y, 0)
    paste(image, sprite, x, y, opacity)


def render_follow(
    video: str | Path,
    trajectory,
    target,
    out_path: str | Path,
    *,
    info: VideoInfo | None = None,
    max_side: int | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> int:
    """Render the followed player over the analysed stretch. Returns frames written."""
    info = info or probe(video)
    if len(trajectory.times) == 0:
        raise ValueError("nothing was analysed")
    start = float(trajectory.times[0])
    end = float(trajectory.times[-1]) + 1.0 / max(info.fps, 1.0)
    rate = min(info.fps, MAX_RENDER_FPS)
    ring_bgr = HIGHLIGHT_RGB[::-1]
    sprite = None
    writer = None
    written = 0
    started = time.perf_counter()
    frames = iter_frames(video, max_side=max_side or DEFAULT_MAX_SIDE, fps=rate, start_s=start,
                         duration_s=end - start, info=info)
    try:
        for frame in frames:
            image = frame.image
            height, width = image.shape[:2]
            if writer is None:
                sprite = label_sprite(target.label, max(14, round(height / 30)))
                writer = VideoWriter(out_path, width, height, rate)
            found = trajectory.box_at(frame.time_s)
            if found is not None:
                box, state = found
                sx, sy = width / info.display_width, height / info.display_height
                scaled = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
                seen = state == "seen"
                draw_ring(image, scaled, ring_bgr, 3.0 if seen else 1.5)
                place_label(image, sprite, scaled, 1.0 if seen else 0.65)
            writer.write(image)
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
    return written


def render_tracks(video: str | Path, tracks, out_path: str | Path, *,
                  info: VideoInfo | None = None, max_side: int | None = None) -> int:
    """Every track with its id, for checking the tracker (R3)."""
    info = info or probe(video)
    by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(tracks.frame):
        by_frame.setdefault(int(frame), []).append(row)
    sprites: dict[tuple[int, int], np.ndarray] = {}
    writer = None
    written = 0
    end = float(tracks.times[-1]) + 1.0 / tracks.fps if len(tracks.times) else tracks.start_s
    frames = iter_frames(video, max_side=max_side or DEFAULT_MAX_SIDE, fps=tracks.fps,
                         start_s=tracks.start_s, duration_s=end - tracks.start_s, info=info)
    try:
        for frame in frames:
            image = frame.image
            height, width = image.shape[:2]
            if writer is None:
                writer = VideoWriter(out_path, width, height, tracks.fps)
            sx, sy = width / info.display_width, height / info.display_height
            font_px = max(11, round(height / 50))
            for row in by_frame.get(frame.index, []):
                track_id = int(tracks.track[row])
                bgr = ID_COLOURS_BGR[track_id % len(ID_COLOURS_BGR)]
                x1, y1, x2, y2 = tracks.boxes[row]
                box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                draw_box(image, box, bgr, 2)
                key = (track_id, font_px)
                if key not in sprites:
                    sprites[key] = label_sprite(str(track_id), font_px, bgr[::-1])
                place_label(image, sprites[key], box)
            writer.write(image)
            written += 1
        if writer is not None:
            writer.close()
            writer = None
    finally:
        frames.close()
        if writer is not None:
            writer.abort()
    return written
