"""The ball's path through the video, from the detector's ball candidates.

Each analysed frame may have zero or several "sports ball" detections (a
head or a bright shoe can look like one). The path keeps the candidate that
is ball-sized compared with the players and reachable from where the ball
just was at kicking speed, preferring confident detections. Gaps of up to
half a second are filled by interpolation. Longer gaps stay unknown: the
ball was hidden, out of shot, or too small to see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A kicked youth ball reaches ~20 m/s: about 14 body heights per second.
MAX_SPEED_HEIGHTS = 15.0
MAX_FILL_S = 0.5
MIN_SCORE = 0.2
# A size-4 ball is ~0.15 of a child's height; allow plenty for blur and boxes.
MAX_SIZE_HEIGHTS = 0.6


@dataclass
class BallPath:
    times: np.ndarray
    xy: np.ndarray  # (n, 2) ball centre, source pixels; NaN where unknown
    state: list[str]  # "seen", "filled" or "unknown"

    def at(self, i: int) -> np.ndarray | None:
        return None if self.state[i] == "unknown" else self.xy[i]

    def seen_fraction(self) -> float:
        return self.state.count("seen") / len(self.state) if self.state else 0.0


def ball_path(times: np.ndarray, fps: float, ball_frame: np.ndarray, ball_boxes: np.ndarray,
              ball_scores: np.ndarray, player_height: float) -> BallPath:
    """`player_height`: a typical player height in pixels, to judge sizes and speeds."""
    n = len(times)
    xy = np.full((n, 2), np.nan)
    state = ["unknown"] * n
    height = max(float(player_height), 1.0)
    candidates: dict[int, list[tuple[np.ndarray, float]]] = {}
    for frame, box, score in zip(ball_frame, ball_boxes, ball_scores, strict=True):
        size = max(box[2] - box[0], box[3] - box[1])
        if score < MIN_SCORE or size > MAX_SIZE_HEIGHTS * height:
            continue
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        candidates.setdefault(int(frame), []).append((centre, float(score)))

    last_i, last_xy = None, None
    for i in range(n):
        options = candidates.get(i)
        if not options:
            continue
        if last_i is None or (i - last_i) / fps > 2.0:
            best = max(options, key=lambda o: o[1])  # fresh start: most confident
        else:
            reach = MAX_SPEED_HEIGHTS * height * (i - last_i) / fps + height
            near = [o for o in options if np.hypot(*(o[0] - last_xy)) <= reach]
            if not near:
                continue  # nothing reachable: probably not the ball
            best = max(near, key=lambda o: o[1] - 0.5 * np.hypot(*(o[0] - last_xy)) / reach)
        xy[i] = best[0]
        state[i] = "seen"
        last_i, last_xy = i, best[0]

    seen = [i for i in range(n) if state[i] == "seen"]
    max_gap = round(MAX_FILL_S * fps)
    for a, b in zip(seen, seen[1:], strict=False):
        if 1 < b - a <= max_gap + 1:
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                xy[i] = xy[a] * (1 - w) + xy[b] * w
                state[i] = "filled"
    return BallPath(times=np.asarray(times, dtype=float), xy=xy, state=state)
