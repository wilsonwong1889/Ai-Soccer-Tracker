"""Generate synthetic sample clips.

Tests and demos use these instead of real footage, so no recording of a child
ever needs to be in the repository. The clip is a green pitch with white
lines, coloured boxes moving like players and a small white "ball". It
exercises the pipeline end to end, but it is not a stand-in for evaluating
detection: real footage is needed for that (see docs/roadmap.md).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .media import require_tool

CODECS = {"h264": "libx264", "hevc": "libx265"}

# (colour, x expression, y expression) as fractions of the frame, over time t.
_MOVERS = [
    ("red", "0.10+0.30*(1+sin(t*0.9))", "0.30+0.10*sin(t*1.7)"),
    ("red", "0.55+0.15*sin(t*1.3)", "0.60+0.15*cos(t*0.8)"),
    ("blue", "0.70-0.25*(1+sin(t*0.7))/2", "0.25+0.20*(1+cos(t*1.1))/2"),
    ("blue", "0.25+0.20*cos(t*1.2)", "0.65+0.10*sin(t*2.1)"),
    ("yellow", "0.05+0.02*sin(t*0.5)", "0.45+0.05*sin(t*0.6)"),
]


def _filter_graph(width: int, height: int, fps: float, duration: float) -> str:
    """A lavfi graph: static pitch, then each mover overlaid at a position over time.

    drawbox can't move (its `t` is thickness), but overlay evaluates x/y
    expressions every frame with `t` as the time in seconds.
    """
    box_w, box_h = max(4, width // 40), max(8, height // 12)
    ball = max(2, width // 120)
    source = f"r={fps}:d={duration}"
    movers = [*_MOVERS, ("white", "0.5+0.35*sin(t*1.9)", "0.5+0.30*cos(t*1.4)")]

    chains = [
        f"color=c=0x2e7d32:s={width}x{height}:{source},"
        # Halfway line and touchlines.
        f"drawbox=x={width // 2 - 1}:y=0:w=3:h={height}:c=white@0.8:t=fill,"
        f"drawbox=x={width // 40}:y={height // 20}:w={width - width // 20}"
        f":h={height - height // 10}:c=white@0.8:t=3[v0]"
    ]
    for i, (colour, x, y) in enumerate(movers):
        size = f"{ball}x{ball}" if i == len(movers) - 1 else f"{box_w}x{box_h}"
        chains.append(f"color=c={colour}:s={size}:{source}[m{i}]")
        chains.append(f"[v{i}][m{i}]overlay=x='({x})*W':y='({y})*H'[v{i + 1}]")
    # lavfi takes the graph's output from the pad labelled out0.
    chains[-1] = chains[-1].rsplit("[", 1)[0] + "[out0]"
    return ";".join(chains)


def make_sample(
    out: str | Path,
    *,
    duration: float = 5.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30,
    codec: str = "h264",
    rotation: int = 0,
    audio: bool = True,
) -> Path:
    """Write a synthetic clip to `out` and return its path.

    `rotation` stores display-rotation metadata (counter-clockwise degrees,
    as iPhones do when held in portrait) without changing the pixels.
    """
    if codec not in CODECS:
        raise ValueError(f"codec must be one of {sorted(CODECS)}")
    if width % 2 or height % 2:
        raise ValueError("width and height must be even")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_tool("ffmpeg")

    command = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", _filter_graph(width, height, fps, duration),
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:d={duration}"]
    command += ["-map", "0:v"]
    if audio:
        command += ["-map", "1:a", "-c:a", "aac", "-b:a", "64k"]
    command += ["-c:v", CODECS[codec], "-pix_fmt", "yuv420p", "-preset", "veryfast"]
    if codec == "hevc":
        # Apple players expect the hvc1 tag; x265 logs a banner unless quietened.
        command += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    command += ["-shortest"]

    if not rotation:
        subprocess.run([*command, str(out)], check=True, capture_output=True)
        return out

    # Rotation is container metadata, set while remuxing (no re-encode).
    # -display_rotation needs FFmpeg 6.0 or newer.
    with tempfile.TemporaryDirectory() as temp:
        plain = Path(temp) / f"plain{out.suffix or '.mp4'}"
        subprocess.run([*command, str(plain)], check=True, capture_output=True)
        subprocess.run(
            [
                ffmpeg, "-v", "error", "-nostdin", "-y",
                "-display_rotation", str(rotation), "-i", str(plain),
                "-map", "0", "-c", "copy", str(out),
            ],
            check=True,
            capture_output=True,
        )
    return out
