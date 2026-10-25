"""Run a detector over a video: detections, an annotated preview and metrics.

The detector sees the small detection frames from `iter_frames`. Every box
is written in *source* pixels (the upright original), so later stages (the
child's crop, exports) can use them at full resolution without rescaling.

Outputs, in the run folder:

    detections.jsonl   one line per analysed frame:
                       {"frame": 0, "t": 0.0, "boxes": [[x1, y1, x2, y2, score, label], ...]}
    annotated.mp4      the detection frames with boxes drawn (unless disabled)
    metrics.json       speed, counts, confidences and box sizes (see run_detection)
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from . import __version__
from .frames import DEFAULT_MAX_SIDE, iter_frames
from .media import VideoInfo, probe
from .render import VideoWriter, draw_boxes

PERSON = "person"
BALL = "sports ball"
DEFAULT_LABELS = (PERSON, BALL)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def scaled(self, sx: float, sy: float) -> Box:
        return Box(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy, self.score, self.label)

    def to_list(self) -> list:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1),
                round(self.score, 4), self.label]


class Detector(Protocol):
    """Anything that finds boxes in a BGR uint8 image, in that image's pixels."""

    name: str

    def detect(self, image: np.ndarray) -> list[Box]: ...

    def describe(self) -> dict: ...


@dataclass
class Progress:
    frames: int
    video_s: float
    total_video_s: float
    elapsed_s: float


def _percentiles(values: list[float], points=(10, 50, 90)) -> dict | None:
    if not values:
        return None
    result = np.percentile(np.asarray(values, dtype=float), points)
    return {f"p{p}": round(float(v), 1) for p, v in zip(points, result, strict=True)}


def run_detection(
    video: str | Path,
    detector: Detector,
    out_dir: str | Path,
    *,
    labels: Iterable[str] = DEFAULT_LABELS,
    max_side: int | None = DEFAULT_MAX_SIDE,
    fps: float | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    write_video: bool = True,
    info: VideoInfo | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> dict:
    """Detect objects through `video` (or a window of it) and write the run folder.

    Returns the metrics, also written to metrics.json. The speed figure the
    roadmap asks for is `realtime_factor`: wall-clock seconds per second of
    video (below 1 is faster than real time).
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = info or probe(video)
    keep = set(labels)
    end_s = info.duration_s if duration_s is None else min(info.duration_s, start_s + duration_s)
    window_s = max(0.0, end_s - start_s)

    frames = iter_frames(
        video, max_side=max_side, fps=fps, start_s=start_s, duration_s=duration_s, info=info
    )
    detect_times: list[float] = []
    counts = {label: 0 for label in sorted(keep)}
    frames_with = {label: 0 for label in sorted(keep)}
    scores = {label: [] for label in sorted(keep)}
    person_heights: list[float] = []
    frame_size = None
    writer = None
    analysed_s = 0.0
    started = time.perf_counter()

    try:
        with (out_dir / "detections.jsonl").open("w") as lines:
            for frame in frames:
                height, width = frame.image.shape[:2]
                if frame_size is None:
                    frame_size = (width, height)
                    if write_video:
                        rate = fps if fps is not None else info.avg_frame_rate
                        writer = VideoWriter(out_dir / "annotated.mp4", width, height, rate)
                t0 = time.perf_counter()
                found = [box for box in detector.detect(frame.image) if box.label in keep]
                detect_times.append(time.perf_counter() - t0)

                sx, sy = info.display_width / width, info.display_height / height
                source_boxes = [box.scaled(sx, sy) for box in found]
                lines.write(json.dumps({
                    "frame": frame.index,
                    "t": round(frame.time_s, 4),
                    "boxes": [box.to_list() for box in source_boxes],
                }) + "\n")

                seen = set()
                for box in source_boxes:
                    counts[box.label] += 1
                    scores[box.label].append(box.score)
                    seen.add(box.label)
                    if box.label == PERSON:
                        person_heights.append(box.height)
                for label in seen:
                    frames_with[label] += 1

                if writer is not None:
                    draw_boxes(frame.image, found)
                    writer.write(frame.image)
                analysed_s = frame.time_s - start_s
                if on_progress is not None:
                    on_progress(Progress(
                        frames=len(detect_times),
                        video_s=analysed_s,
                        total_video_s=window_s,
                        elapsed_s=time.perf_counter() - started,
                    ))
        if writer is not None:
            writer.close()
            writer = None
    finally:
        frames.close()
        if writer is not None:
            writer.abort()  # an error is on its way up: drop the partial video

    wall_s = time.perf_counter() - started
    n = len(detect_times)
    times_ms = [t * 1000 for t in detect_times]
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "source": {
            "filename": info.filename,
            "duration_s": round(info.duration_s, 3),
            "display_size": [info.display_width, info.display_height],
            "fps": round(info.fps, 3),
        },
        "window": {"start_s": round(start_s, 3), "seconds": round(window_s, 3)},
        "settings": {
            "max_side": max_side,
            "fps": fps if fps is not None else round(info.fps, 3),
            "detection_frame_size": list(frame_size) if frame_size else None,
            "labels": sorted(keep),
        },
        "detector": detector.describe(),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or None},
        "frames": n,
        "speed": {
            "wall_s": round(wall_s, 2),
            "realtime_factor": round(wall_s / window_s, 3) if window_s else None,
            "frames_per_s": round(n / wall_s, 2) if wall_s else None,
            "detect_ms": _percentiles(times_ms, (50, 90, 99)),
            "detect_share": round(sum(detect_times) / wall_s, 3) if wall_s else None,
        },
        "detections": {
            label: {
                "total": counts[label],
                "per_frame": round(counts[label] / n, 2) if n else 0.0,
                "frames_with_any": round(frames_with[label] / n, 3) if n else 0.0,
                "mean_score": round(float(np.mean(scores[label])), 3) if scores[label] else None,
            }
            for label in sorted(keep)
        },
        # How tall players are in source pixels: small players are the main
        # detection risk in a wide sideline shot.
        "person_height_px": _percentiles(person_heights),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def read_detections(path: str | Path) -> list[dict]:
    """Load detections.jsonl back as [{"frame", "t", "boxes": [Box, ...]}, ...]."""
    frames = []
    with Path(path).open() as lines:
        for line in lines:
            row = json.loads(line)
            row["boxes"] = [Box(*values) for values in row["boxes"]]
            frames.append(row)
    return frames
