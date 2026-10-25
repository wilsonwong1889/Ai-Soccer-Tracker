"""Track every player through a video (R3).

Detection runs on each analysed frame. ByteTrack, from Roboflow's
Apache-2.0 `trackers` package, links the boxes into tracks, and each
tracked box gets an appearance description (appearance.py). Everything is
saved, so following a different player later doesn't need detection again:

    tracks.npz     frame index, time, track id, box (source pixels), score,
                   appearance, one row per tracked box
    tracks.txt     the same tracks in MOT format, for scoring with TrackEval
    metrics.json   speed and track counts
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .appearance import FEATURE_SIZE, describe, mean_feature, similarity
from .detect import PERSON, Detector, Progress
from .frames import DEFAULT_MAX_SIDE, iter_frames
from .media import VideoInfo, probe

DEFAULT_TRACK_FPS = 15.0
# Seconds a lost track is kept alive for ByteTrack to pick it up again.
LOST_BUFFER_S = 1.0


@dataclass
class Tracks:
    """All tracked boxes of one tracking run, in source-video pixels."""

    fps: float
    start_s: float
    times: np.ndarray  # time of each analysed frame
    frame: np.ndarray  # (n,) analysed-frame index of each box
    track: np.ndarray  # (n,) track id
    boxes: np.ndarray  # (n, 4) x1, y1, x2, y2
    scores: np.ndarray  # (n,)
    features: np.ndarray  # (n, FEATURE_SIZE); NaN rows where too small to describe
    size: tuple[int, int] = (0, 0)  # source display width, height

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, fps=self.fps, start_s=self.start_s, times=self.times, frame=self.frame,
            track=self.track, boxes=self.boxes, scores=self.scores, features=self.features,
            size=np.array(self.size),
        )

    @classmethod
    def load(cls, path: Path) -> Tracks:
        data = np.load(path)
        return cls(
            fps=float(data["fps"]), start_s=float(data["start_s"]), times=data["times"],
            frame=data["frame"], track=data["track"], boxes=data["boxes"],
            scores=data["scores"], features=data["features"],
            size=tuple(int(v) for v in data["size"]),
        )

    def feature(self, row: int) -> np.ndarray | None:
        f = self.features[row]
        return None if np.isnan(f[0]) else f


@dataclass
class Tracklet:
    """One continuous stretch of one player, as ByteTrack (or a split) saw it."""

    id: int
    frames: np.ndarray  # analysed-frame indices, increasing
    boxes: np.ndarray  # (k, 4)
    scores: np.ndarray
    features: list = field(repr=False, default_factory=list)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def appearance(self, around: int | None = None, window: int | None = None):
        """Mean appearance, optionally only near analysed frame `around`."""
        if around is None or window is None:
            return mean_feature(self.features)
        keep = [f for fr, f in zip(self.frames, self.features, strict=True)
                if abs(int(fr) - around) <= window]
        return mean_feature(keep) if any(f is not None for f in keep) else mean_feature(
            self.features)

    def centers(self) -> np.ndarray:
        return np.stack([(self.boxes[:, 0] + self.boxes[:, 2]) / 2,
                         (self.boxes[:, 1] + self.boxes[:, 3]) / 2], axis=1)

    def heights(self) -> np.ndarray:
        return self.boxes[:, 3] - self.boxes[:, 1]


def tracklets(tracks: Tracks) -> list[Tracklet]:
    result = []
    for track_id in np.unique(tracks.track):
        rows = np.flatnonzero(tracks.track == track_id)
        rows = rows[np.argsort(tracks.frame[rows], kind="stable")]
        result.append(Tracklet(
            id=int(track_id), frames=tracks.frame[rows], boxes=tracks.boxes[rows],
            scores=tracks.scores[rows], features=[tracks.feature(r) for r in rows],
        ))
    return result


def split_on_appearance(items: list[Tracklet], fps: float, threshold: float = 0.55,
                        window_s: float = 0.5) -> list[Tracklet]:
    """Cut tracklets where the player's colours change abruptly.

    When two players cross, a tracker can carry one player's id onto the
    other. If they're on different teams, the colours change at that moment;
    cutting there lets the follower choose the right continuation. (Two
    teammates swapping is invisible to this: that needs jersey numbers.)
    """
    window = max(2, round(window_s * fps))
    result = []
    next_id = max((t.id for t in items), default=0) + 1
    for item in items:
        n = len(item.frames)
        cuts = []
        if n >= 2 * window:
            scores = []
            for i in range(window, n - window + 1):
                before = mean_feature(item.features[i - window:i])
                after = mean_feature(item.features[i:i + window])
                scores.append((similarity(before, after), i))
            # Lowest-similarity points first, at least a window apart.
            for value, i in sorted(scores):
                if value >= threshold:
                    break
                if all(abs(i - c) >= window for c in cuts):
                    cuts.append(i)
        bounds = [0, *sorted(cuts), n]
        for part, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
            piece_id = item.id if part == 0 else next_id
            if part:
                next_id += 1
            result.append(Tracklet(piece_id, item.frames[a:b], item.boxes[a:b],
                                   item.scores[a:b], item.features[a:b]))
    return result


def run_tracking(
    video: str | Path,
    detector: Detector,
    out_dir: str | Path,
    *,
    fps: float | None = DEFAULT_TRACK_FPS,
    max_side: int | None = DEFAULT_MAX_SIDE,
    start_s: float = 0.0,
    duration_s: float | None = None,
    info: VideoInfo | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> Tracks:
    """Detect and track players through `video`, saving the run folder."""
    from supervision import Detections
    from trackers import ByteTrackTracker

    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = info or probe(video)
    rate = min(fps, info.fps) if fps else info.fps
    end_s = info.duration_s if duration_s is None else min(info.duration_s, start_s + duration_s)
    window_s = max(0.0, end_s - start_s)
    tracker = ByteTrackTracker(
        lost_track_buffer=max(1, round(LOST_BUFFER_S * rate)),
        frame_rate=rate,
        track_activation_threshold=0.4,
        minimum_consecutive_frames=2,
        high_conf_det_threshold=0.5,
    )

    times, frame_ix, track_ids, boxes, scores, features = [], [], [], [], [], []
    started = time.perf_counter()
    frame_size = None
    for frame in iter_frames(video, max_side=max_side, fps=rate, start_s=start_s,
                             duration_s=duration_s, info=info):
        image = frame.image
        height, width = image.shape[:2]
        frame_size = (width, height)
        people = [b for b in detector.detect(image) if b.label == PERSON]
        detections = Detections(
            xyxy=np.array([[b.x1, b.y1, b.x2, b.y2] for b in people], dtype=float).reshape(-1, 4),
            confidence=np.array([b.score for b in people], dtype=float),
            class_id=np.zeros(len(people), dtype=int),
        )
        tracked = tracker.update(detections)
        sx, sy = info.display_width / width, info.display_height / height
        times.append(frame.time_s)
        for xyxy, score, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id,
                                         strict=True):
            if track_id < 0:
                continue  # not confirmed yet
            feature = describe(image, tuple(xyxy))
            frame_ix.append(frame.index)
            track_ids.append(int(track_id))
            boxes.append([xyxy[0] * sx, xyxy[1] * sy, xyxy[2] * sx, xyxy[3] * sy])
            scores.append(float(score))
            features.append(feature if feature is not None
                            else np.full(FEATURE_SIZE, np.nan, dtype=np.float32))
        if on_progress is not None:
            on_progress(Progress(frames=len(times), video_s=frame.time_s - start_s,
                                 total_video_s=window_s,
                                 elapsed_s=time.perf_counter() - started))

    result = Tracks(
        fps=rate, start_s=start_s, times=np.array(times, dtype=float),
        frame=np.array(frame_ix, dtype=int), track=np.array(track_ids, dtype=int),
        boxes=np.array(boxes, dtype=float).reshape(-1, 4),
        scores=np.array(scores, dtype=float),
        features=np.array(features, dtype=np.float32).reshape(-1, FEATURE_SIZE),
        size=(info.display_width, info.display_height),
    )
    result.save(out_dir / "tracks.npz")
    write_mot(result, out_dir / "tracks.txt")
    lengths = [len(t.frames) / rate for t in tracklets(result)]
    wall_s = time.perf_counter() - started
    metrics = {
        "kind": "tracks",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "source": {"filename": info.filename, "duration_s": round(info.duration_s, 3)},
        "window": {"start_s": round(start_s, 3), "seconds": round(window_s, 3)},
        "settings": {"fps": rate, "max_side": max_side,
                     "detection_frame_size": list(frame_size) if frame_size else None},
        "detector": detector.describe(),
        "frames": len(times),
        "tracks": {
            "count": len(lengths),
            "median_length_s": round(float(np.median(lengths)), 2) if lengths else None,
            "per_frame": round(len(track_ids) / len(times), 2) if times else 0.0,
        },
        "speed": {"wall_s": round(wall_s, 2),
                  "realtime_factor": round(wall_s / window_s, 3) if window_s else None},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return result


def tracks_dir(runs_base: Path, detector_name: str, fps: float, start_s: float,
               duration_s: float | None) -> Path:
    """Where a tracking run with these settings is kept, under a video's runs folder."""
    window = "all" if duration_s is None else f"{duration_s:g}s"
    return Path(runs_base) / f"tracks-{detector_name}-{fps:g}fps-from{start_s:g}s-{window}"


def load_or_track(video: str | Path, runs_base: Path, detector: Detector, *,
                  fps: float = DEFAULT_TRACK_FPS, start_s: float = 0.0,
                  duration_s: float | None = None, max_side: int | None = DEFAULT_MAX_SIDE,
                  fresh: bool = False,
                  on_progress: Callable[[Progress], None] | None = None,
                  ) -> tuple[Tracks, Path, bool]:
    """(tracks, folder, reused): reuses an earlier run with the same settings.

    Tracking is the slow part, so following a second player in the same
    stretch of the same video costs only the following and the drawing.
    """
    folder = tracks_dir(runs_base, detector.name, fps, start_s, duration_s)
    if not fresh and (folder / "tracks.npz").is_file() and (folder / "metrics.json").is_file():
        return Tracks.load(folder / "tracks.npz"), folder, True
    tracks = run_tracking(video, detector, folder, fps=fps, max_side=max_side, start_s=start_s,
                          duration_s=duration_s, on_progress=on_progress)
    return tracks, folder, False


def write_mot(tracks: Tracks, path: Path) -> None:
    """MOTChallenge text: frame (1-based), id (1-based), x, y, w, h, score, -1, -1, -1."""
    with path.open("w") as out:
        for i in np.argsort(tracks.frame, kind="stable"):
            x1, y1, x2, y2 = tracks.boxes[i]
            out.write(f"{tracks.frame[i] + 1},{tracks.track[i] + 1},{x1:.1f},{y1:.1f},"
                      f"{x2 - x1:.1f},{y2 - y1:.1f},{tracks.scores[i]:.3f},-1,-1,-1\n")
