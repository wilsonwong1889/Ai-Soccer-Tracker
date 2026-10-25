"""Follow one player through a tracked video: the child the parent tapped.

1. The tap picks the tracklet under it, at that moment.
2. Forwards and backwards from there, whenever the tracklet ends (the
   player was hidden, or the tracker lost them), the next piece of the same
   player is chosen among tracklets that start soon after. They must start
   where the player could have run to in that time, be about the same size,
   and look like the player did when tapped. Each link is scored and saved;
   close calls are marked uncertain.
3. Short gaps are filled in by interpolation, and the path is smoothed so the
   highlight and label glide instead of jittering.

The jersey number is carried through and shown in the label. Reading it from
the video to confirm (or find) the player is R5.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .appearance import similarity
from .detect import Progress
from .tracking import Tracklet, Tracks, split_on_appearance, tracklets


class SeedNotFound(ValueError):
    """No player where the parent tapped."""


@dataclass(frozen=True)
class Seed:
    t: float  # seconds into the video
    x: float  # source-video pixels
    y: float


@dataclass(frozen=True)
class Target:
    number: str
    name: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}  #{self.number}" if self.name else f"#{self.number}"


@dataclass(frozen=True)
class Params:
    max_gap_s: float = 12.0  # longest hidden stretch to reconnect across
    short_gap_s: float = 2.0  # gaps up to this trust motion more
    fill_gap_s: float = 3.0  # gaps up to this are drawn by interpolation
    base_radius: float = 1.5  # body heights the player may be from where expected...
    speed: float = 4.0  # ...plus this many body heights per second hidden
    min_appearance: float = 0.55  # colour similarity needed after a short gap
    min_appearance_long: float = 0.7  # ...and after a long one
    appearance_weight: float = 0.6  # vs. motion, in a link's score
    uncertain_margin: float = 0.05  # runner-up this close: the link is uncertain
    smooth_center_s: float = 0.15  # Gaussian smoothing of position
    smooth_size_s: float = 0.5  # ...and of box size


DEFAULT_PARAMS = Params()


@dataclass
class Link:
    from_id: int
    to_id: int
    direction: str
    gap_s: float
    score: float
    appearance: float
    motion: float
    uncertain: bool


@dataclass
class Trajectory:
    """The followed player's box at each analysed frame (NaN where lost)."""

    times: np.ndarray
    boxes: np.ndarray
    state: list[str]  # "seen", "filled" (interpolated) or "lost"
    links: list[Link] = field(default_factory=list)

    def box_at(self, t: float) -> tuple[np.ndarray, str] | None:
        """Box at any time, interpolated between analysed frames; None if lost."""
        times = self.times
        if len(times) == 0 or t < times[0] - 1e-6 or t > times[-1] + 1e-6:
            return None
        i = int(np.searchsorted(times, t, side="right")) - 1
        i = min(max(i, 0), len(times) - 1)
        if self.state[i] == "lost":
            return None
        if i == len(times) - 1 or abs(t - times[i]) < 1e-6:
            return self.boxes[i], self.state[i]
        if self.state[i + 1] == "lost":
            return self.boxes[i], self.state[i]
        w = (t - times[i]) / (times[i + 1] - times[i])
        state = "seen" if self.state[i] == self.state[i + 1] == "seen" else "filled"
        return self.boxes[i] * (1 - w) + self.boxes[i + 1] * w, state


def _index(item: Tracklet, frame: int) -> int | None:
    hits = np.flatnonzero(item.frames == frame)
    return int(hits[0]) if len(hits) else None


def find_seed(items: list[Tracklet], times: np.ndarray, seed: Seed) -> tuple[Tracklet, int]:
    """The tracklet under the tap, and the analysed frame used."""
    if len(times) == 0:
        raise SeedNotFound("No frames were analysed.")
    if seed.t < times[0] - 0.5 or seed.t > times[-1] + 0.5:
        raise SeedNotFound(
            f"The moment you picked ({seed.t:.1f} s) is outside the part that was analysed "
            f"({times[0]:.1f}–{times[-1]:.1f} s)."
        )
    order = np.argsort(np.abs(times - seed.t))
    # Look at the nearest few analysed frames in case the player was missed in one.
    for frame in order[:5]:
        inside, near = [], []
        for item in items:
            i = _index(item, int(frame))
            if i is None:
                continue
            x1, y1, x2, y2 = item.boxes[i]
            cx, cy, h = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0)
            distance = float(np.hypot(seed.x - cx, seed.y - cy))
            if x1 <= seed.x <= x2 and y1 <= seed.y <= y2:
                inside.append((distance, item))
            elif distance <= 0.75 * h:
                near.append((distance / h, item))
        best = min(inside, key=lambda d: d[0], default=None) or min(
            near, key=lambda d: d[0], default=None)
        if best is not None:
            return best[1], int(frame)
    raise SeedNotFound(
        f"No player was found where you tapped at {seed.t:.1f} s. "
        "Pick a moment where your child is clearly visible and tap the middle of them."
    )


def _end_state(item: Tracklet, fps: float, forward: bool, span_s: float = 0.5):
    """Position, velocity (pixels per analysed frame) and height at one end."""
    centers, heights = item.centers(), item.heights()
    k = min(len(item.frames), max(2, round(span_s * fps)))
    if forward:
        idx_a, idx_b = len(item.frames) - k, len(item.frames) - 1
    else:
        idx_a, idx_b = 0, k - 1
    span = max(int(item.frames[idx_b] - item.frames[idx_a]), 1)
    velocity = (centers[idx_b] - centers[idx_a]) / span if k >= 2 else np.zeros(2)
    end = idx_b if forward else idx_a
    # Cap the speed: a short noisy run shouldn't fling the prediction across the pitch.
    limit = 6.0 * heights[end] / fps
    speed = float(np.hypot(*velocity))
    if speed > limit:
        velocity = velocity * (limit / speed)
    return centers[end], velocity, float(heights[end])


def _candidates(current: Tracklet, pool: list[Tracklet], forward: bool, fps: float,
                model, params: Params) -> list[tuple[float, Tracklet, float, float, float]]:
    center, velocity, height = _end_state(current, fps, forward)
    scored = []
    for cand in pool:
        gap = (cand.start - current.end) if forward else (current.start - cand.end)
        if gap < 1:
            continue
        gap_s = gap / fps
        if gap_s > params.max_gap_s:
            continue
        predicted = center + velocity * gap if forward else center - velocity * gap
        c_center = cand.centers()[0 if forward else -1]
        c_height = float(cand.heights()[0 if forward else -1])
        if not 0.5 <= c_height / max(height, 1.0) <= 2.0:
            continue
        distance = float(np.hypot(*(c_center - predicted))) / max(height, 1.0)
        radius = params.base_radius + params.speed * gap_s
        if distance > radius:
            continue
        motion = 1.0 - distance / radius
        look = similarity(model, cand.appearance())
        needed = params.min_appearance if gap_s <= params.short_gap_s else (
            params.min_appearance_long)
        if look < needed:
            continue
        score = params.appearance_weight * look + (1 - params.appearance_weight) * motion
        scored.append((score, cand, gap_s, look, motion))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored


def link_chain(items: list[Tracklet], seed_item: Tracklet, seed_frame: int, fps: float,
               params: Params) -> tuple[list[Tracklet], list[Link]]:
    """The seed tracklet plus the pieces linked to it, in time order."""
    model = seed_item.appearance(around=seed_frame, window=round(fps))
    pool = [t for t in items if t is not seed_item]
    chain, links = [seed_item], []
    for forward in (True, False):
        current = seed_item
        while True:
            options = _candidates(current, pool, forward, fps, model, params)
            if not options:
                break
            score, chosen, gap_s, look, motion = options[0]
            uncertain = len(options) > 1 and options[1][0] >= score - params.uncertain_margin
            links.append(Link(current.id, chosen.id, "forward" if forward else "backward",
                              round(gap_s, 2), round(score, 3), round(look, 3),
                              round(motion, 3), uncertain))
            pool.remove(chosen)
            # Nothing else may overlap the chosen piece in time.
            pool = [t for t in pool if t.end < chosen.start or t.start > chosen.end]
            chain.append(chosen)
            current = chosen
    chain.sort(key=lambda t: t.start)
    return chain, links


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 3:
        return values
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    padded = np.pad(values, (radius, radius))
    weights = np.pad(np.ones_like(values), (radius, radius))
    num = np.convolve(padded, kernel, mode="valid")
    den = np.convolve(weights, kernel, mode="valid")
    return num / den  # renormalised at the ends, so no pull towards zero


def build_trajectory(times: np.ndarray, chain: list[Tracklet], fps: float,
                     params: Params) -> Trajectory:
    n = len(times)
    boxes = np.full((n, 4), np.nan)
    state = ["lost"] * n
    for item in chain:
        for frame, box in zip(item.frames, item.boxes, strict=True):
            boxes[frame] = box
            state[frame] = "seen"

    # Fill short gaps, between pieces or inside one, by straight-line interpolation.
    seen = [i for i in range(n) if state[i] == "seen"]
    max_fill = round(params.fill_gap_s * fps)
    for a, b in zip(seen, seen[1:], strict=False):
        if 1 < b - a <= max_fill + 1:
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                boxes[i] = boxes[a] * (1 - w) + boxes[b] * w
                state[i] = "filled"

    # Smooth each continuous stretch: centre lightly, size more.
    i = 0
    while i < n:
        if state[i] == "lost":
            i += 1
            continue
        j = i
        while j + 1 < n and state[j + 1] != "lost":
            j += 1
        run = boxes[i:j + 1]
        cx, cy = (run[:, 0] + run[:, 2]) / 2, (run[:, 1] + run[:, 3]) / 2
        w, h = run[:, 2] - run[:, 0], run[:, 3] - run[:, 1]
        sigma = params.smooth_center_s * fps
        cx, cy = _smooth(cx, sigma), _smooth(cy, sigma)
        w, h = _smooth(w, params.smooth_size_s * fps), _smooth(h, params.smooth_size_s * fps)
        boxes[i:j + 1] = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        i = j + 1
    return Trajectory(times=np.asarray(times, dtype=float), boxes=boxes, state=state)


def follow(tracks: Tracks, seed: Seed, params: Params = DEFAULT_PARAMS) -> Trajectory:
    items = split_on_appearance(tracklets(tracks), tracks.fps)
    seed_item, seed_frame = find_seed(items, tracks.times, seed)
    chain, links = link_chain(items, seed_item, seed_frame, tracks.fps, params)
    trajectory = build_trajectory(tracks.times, chain, tracks.fps, params)
    trajectory.links = links
    return trajectory


def follow_player(
    video: str | Path,
    tracks: Tracks,
    seed: Seed,
    target: Target,
    out_dir: str | Path,
    *,
    params: Params = DEFAULT_PARAMS,
    render: bool = True,
    max_side: int | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> dict:
    """Follow the tapped player and write target.json, followed.mp4 and metrics.json."""
    from .media import probe
    from .overlay import render_follow

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trajectory = follow(tracks, seed, params)
    n = len(trajectory.state)
    counts = {s: trajectory.state.count(s) for s in ("seen", "filled", "lost")}
    located = [i for i, s in enumerate(trajectory.state) if s != "lost"]
    (out_dir / "target.json").write_text(json.dumps({
        "target": asdict(target),
        "seed": asdict(seed),
        "fps": tracks.fps,
        "links": [asdict(link) for link in trajectory.links],
        "frames": [
            {"t": round(float(t), 3), "state": s,
             "box": None if s == "lost" else [round(float(v), 1) for v in b]}
            for t, s, b in zip(trajectory.times, trajectory.state, trajectory.boxes, strict=True)
        ],
    }) + "\n")

    info = probe(video)
    if render:
        render_follow(video, trajectory, target, out_dir / "followed.mp4", info=info,
                      max_side=max_side, on_progress=on_progress)
    wall_s = time.perf_counter() - started
    metrics = {
        "kind": "follow",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "target": asdict(target),
        "seed": asdict(seed),
        "source": {"filename": info.filename, "duration_s": round(info.duration_s, 3)},
        "window": {"start_s": round(float(trajectory.times[0]), 3) if n else 0.0,
                   "seconds": round(float(trajectory.times[-1] - trajectory.times[0]), 3)
                   if n else 0.0},
        "frames": n,
        "coverage": {s: round(c / n, 3) if n else 0.0 for s, c in counts.items()},
        "first_seen_s": round(float(trajectory.times[located[0]]), 2) if located else None,
        "last_seen_s": round(float(trajectory.times[located[-1]]), 2) if located else None,
        "links": len(trajectory.links),
        "uncertain_links": sum(link.uncertain for link in trajectory.links),
        "params": asdict(params),
        "speed": {"wall_s": round(wall_s, 2)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
