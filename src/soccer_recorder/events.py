"""What the followed player did: on the ball, off the ball, passes, dribbles, shots.

Everything here is a rule over positions, not a trained action model, so
every event is a *candidate* with a confidence. The confidence mostly says
how much of the moment the ball was actually seen rather than guessed.
Parents confirm or reject events on the website, and those reviews are how
the rules will be measured and improved (R7).

Distances and speeds are measured in the player's own body heights per
frame. That roughly cancels out perspective (players far from the camera
look smaller), and is converted to metres with the player's height,
1.4 m unless the parent gives it. They are estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ball import BallPath
from .follow import Trajectory
from .teams import OTHER, Teams
from .tracking import Tracks

TARGET = "target"


@dataclass(frozen=True)
class EventParams:
    reach: float = 0.6  # ball within this many body heights of the feet: on the ball
    min_spell_s: float = 0.2  # shorter contact is noise
    spell_gap_s: float = 0.4  # brief loss of the ball doesn't end a spell
    pass_gap_s: float = 3.0  # the ball must reach the next player within this
    min_pass_heights: float = 1.5  # ...from a player at least this far away
    dribble_s: float = 1.0  # a spell this long...
    dribble_heights: float = 1.5  # ...covering this distance (~2 m) is a dribble
    shot_speed: float = 6.0  # body heights per second towards a goal
    shot_window_s: float = 1.5
    sprint_mps: float = 4.5  # ~16 km/h: fast for a youth player
    sprint_s: float = 1.0
    player_height_m: float = 1.4


@dataclass
class Event:
    type: str
    t: float
    end_t: float | None = None
    confidence: float = 0.5
    details: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.type}-{round(self.t * 10)}"

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "t": round(self.t, 2),
                "end_t": None if self.end_t is None else round(self.end_t, 2),
                "confidence": round(self.confidence, 2), "details": self.details}


@dataclass
class Spell:
    owner: str | int  # TARGET or a track id
    start: int  # analysed-frame indices, inclusive
    end: int


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def possession(tracks: Tracks, trajectory: Trajectory, ball: BallPath,
               params: EventParams) -> list[str | int | None]:
    """Who has the ball at each analysed frame: TARGET, a track id, or None."""
    n = len(trajectory.times)
    rows_by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(tracks.frame):
        rows_by_frame.setdefault(int(frame), []).append(row)
    owners: list[str | int | None] = [None] * n
    for i in range(n):
        b = ball.at(i)
        if b is None:
            continue
        candidates = []
        target_box = trajectory.boxes[i] if trajectory.state[i] != "lost" else None
        if target_box is not None:
            candidates.append((TARGET, target_box))
        for row in rows_by_frame.get(i, []):
            box = tracks.boxes[row]
            if target_box is not None and _iou(box, target_box) > 0.5:
                continue  # that's the followed player
            candidates.append((int(tracks.track[row]), box))
        best, best_distance = None, params.reach
        for who, (x1, y1, x2, y2) in candidates:
            height = max(y2 - y1, 1.0)
            if b[1] < y1 + 0.45 * height:
                continue  # ball at chest height or above: not at their feet
            distance = np.hypot(b[0] - (x1 + x2) / 2, b[1] - y2) / height
            if distance <= best_distance:
                best, best_distance = who, distance
        owners[i] = best
    return owners


def spells(owners: list, fps: float, params: EventParams) -> list[Spell]:
    """Runs of the same owner, bridging short losses of the ball."""
    result: list[Spell] = []
    max_gap = round(params.spell_gap_s * fps)
    for i, owner in enumerate(owners):
        if owner is None:
            continue
        if result and result[-1].owner == owner and i - result[-1].end <= max_gap + 1:
            result[-1].end = i
        else:
            result.append(Spell(owner, i, i))
    min_frames = max(1, round(params.min_spell_s * fps))
    return [s for s in result if s.end - s.start + 1 >= min_frames]


def _feet(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, box[3]])


def _seen_share(ball: BallPath, a: int, b: int) -> float:
    states = ball.state[max(0, a):b + 1]
    return states.count("seen") / len(states) if states else 0.0


def movement(trajectory: Trajectory, on_ball: np.ndarray, params: EventParams,
             frame_size: tuple[int, int], grid: tuple[int, int] = (24, 14)) -> dict:
    """Distance, speed, sprints and a position heatmap for the followed player."""
    times, boxes, state = trajectory.times, trajectory.boxes, trajectory.state
    n = len(times)
    fps = 1.0 / float(np.median(np.diff(times))) if n > 1 else 1.0
    visible = np.array([s != "lost" for s in state])
    step_m = np.zeros(n)
    for i in range(1, n):
        if visible[i] and visible[i - 1]:
            height = (boxes[i, 3] - boxes[i, 1] + boxes[i - 1, 3] - boxes[i - 1, 1]) / 2
            step = np.hypot(*(_feet(boxes[i]) - _feet(boxes[i - 1]))) / max(height, 1.0)
            step_m[i] = step * params.player_height_m
    # Speed over a one-second window, so a jittery box doesn't count as sprinting.
    window = max(1, round(fps))
    speed = np.convolve(step_m * fps, np.ones(window) / window, mode="same")
    speed[~visible] = 0.0
    fast = speed >= params.sprint_mps
    sprints, run = 0, 0
    for is_fast in np.append(fast, False):
        if is_fast:
            run += 1
        else:
            sprints += run >= params.sprint_s * fps
            run = 0
    width, height = frame_size
    heat = np.zeros((grid[1], grid[0]))
    for i in np.flatnonzero(visible):
        fx, fy = _feet(boxes[i])
        gx = min(grid[0] - 1, max(0, int(fx / max(width, 1) * grid[0])))
        gy = min(grid[1] - 1, max(0, int(fy / max(height, 1) * grid[1])))
        heat[gy, gx] += 1
    total = heat.max() or 1.0
    on = on_ball & visible
    return {
        "visible_s": round(float(visible.sum()) / fps, 1),
        "on_ball_s": round(float(on.sum()) / fps, 1),
        "off_ball_s": round(float((visible & ~on_ball).sum()) / fps, 1),
        "distance_m": round(float(step_m.sum()), 1),
        "distance_on_ball_m": round(float(step_m[on].sum()), 1),
        "distance_off_ball_m": round(float(step_m[visible & ~on_ball].sum()), 1),
        "top_speed_kmh": round(float(speed.max()) * 3.6, 1) if n else 0.0,
        "sprints": int(sprints),
        "heatmap": {"cols": grid[0], "rows": grid[1],
                    "values": np.round(heat / total, 3).tolist()},
        "speed_series": [round(float(v) * 3.6, 1) for v in speed[::max(1, window // 2)]],
    }


def detect_events(tracks: Tracks, trajectory: Trajectory, ball: BallPath, teams: Teams,
                  target_team: int, goals: list[tuple[float, float, float, float]],
                  params: EventParams) -> tuple[list[Event], dict]:
    """Events and statistics for the followed player."""
    fps = tracks.fps
    times = trajectory.times
    owners = possession(tracks, trajectory, ball, params)
    all_spells = spells(owners, fps, params)
    events: list[Event] = []

    def team(owner) -> int:
        return target_team if owner == TARGET else teams.of_track.get(owner, OTHER)

    def height_at(i: int) -> float:
        box = trajectory.boxes[i] if trajectory.state[i] != "lost" else None
        if box is not None:
            return float(box[3] - box[1])
        if not len(tracks.boxes):
            return 1.0
        return float(np.median(tracks.boxes[:, 3] - tracks.boxes[:, 1]))

    rows_by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(tracks.frame):
        rows_by_frame.setdefault(int(frame), []).append(row)

    def feet_of(owner, i: int) -> np.ndarray | None:
        if owner == TARGET:
            return None if trajectory.state[i] == "lost" else _feet(trajectory.boxes[i])
        for row in rows_by_frame.get(i, []):
            if int(tracks.track[row]) == owner:
                return _feet(tracks.boxes[row])
        return None

    on_ball = np.zeros(len(times), dtype=bool)
    for spell in all_spells:
        if spell.owner != TARGET:
            continue
        on_ball[spell.start:spell.end + 1] = True
        seen = _seen_share(ball, spell.start, spell.end)
        duration = (spell.end - spell.start + 1) / fps
        events.append(Event("touch", float(times[spell.start]), float(times[spell.end]),
                            0.35 + 0.6 * seen, {"seconds": round(duration, 1)}))
        start_feet = _feet(trajectory.boxes[spell.start])
        end_feet = _feet(trajectory.boxes[spell.end])
        travelled = np.hypot(*(end_feet - start_feet)) / max(height_at(spell.start), 1.0)
        if duration >= params.dribble_s and travelled >= params.dribble_heights:
            events.append(Event(
                "dribble", float(times[spell.start]), float(times[spell.end]), 0.3 + 0.6 * seen,
                {"seconds": round(duration, 1),
                 "distance_m": round(float(travelled) * params.player_height_m, 1)}))

    max_gap = round(params.pass_gap_s * fps)
    for first, second in zip(all_spells, all_spells[1:], strict=False):
        if first.owner == second.owner or TARGET not in (first.owner, second.owner):
            continue
        if second.start - first.end > max_gap:
            continue
        # How far apart the two players were: close together is a tussle, not a pass.
        a, b = feet_of(first.owner, first.end), feet_of(second.owner, second.start)
        if a is None or b is None:
            continue
        distance = np.hypot(*(b - a)) / max(height_at(first.end), 1.0)
        if distance < params.min_pass_heights:
            continue
        seen = _seen_share(ball, first.end, second.start)
        same = team(first.owner) == team(second.owner) != OTHER
        known = OTHER not in (team(first.owner), team(second.owner))
        if first.owner == TARGET:
            kind = "pass_completed" if same else ("pass_lost" if known else "pass")
        else:
            kind = "pass_received" if same else ("ball_won" if known else "ball_gained")
        events.append(Event(kind, float(times[first.end]), float(times[second.start]),
                            0.3 + 0.5 * seen + (0.15 if known else 0.0),
                            {"distance_m": round(float(distance) * params.player_height_m, 1)}))

    # Shots: the ball leaves the followed player fast, towards a goal.
    window = round(params.shot_window_s * fps)
    for spell in (s for s in all_spells if s.owner == TARGET):
        for k, goal in enumerate(goals):
            centre = np.array([(goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2])
            start = ball.at(spell.end)
            if start is None:
                continue
            for j in range(spell.end + 1, min(len(times), spell.end + window + 1)):
                now = ball.at(j)
                if now is None:
                    continue
                elapsed = (j - spell.end) / fps
                speed = np.hypot(*(now - start)) / max(height_at(spell.end), 1.0) / elapsed
                closer = np.hypot(*(centre - now)) < 0.7 * np.hypot(*(centre - start))
                if speed >= params.shot_speed and closer:
                    events.append(Event("shot", float(times[spell.end]), float(times[j]),
                                        0.3 + 0.5 * _seen_share(ball, spell.end, j),
                                        {"goal": k}))
                    break

    # The ball inside a marked goal.
    for k, (x1, y1, x2, y2) in enumerate(goals):
        inside = [i for i in range(len(times))
                  if (p := ball.at(i)) is not None and x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
        runs: list[list[int]] = []
        for i in inside:
            if runs and i - runs[-1][-1] <= round(fps):
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            if len(run) >= 2 and any(ball.state[i] == "seen" for i in run):
                events.append(Event("ball_in_goal", float(times[run[0]]), float(times[run[-1]]),
                                    0.55, {"goal": k}))

    move = movement(trajectory, on_ball, params, tracks.size)
    counts = {kind: sum(e.type == kind for e in events) for kind in (
        "touch", "dribble", "pass_completed", "pass_lost", "pass", "pass_received",
        "ball_won", "ball_gained", "shot", "ball_in_goal")}
    passes_made = counts["pass_completed"] + counts["pass_lost"] + counts["pass"]
    stats = {
        **move,
        "touches": counts["touch"],
        "dribbles": counts["dribble"],
        "passes_made": passes_made,
        "passes_completed": counts["pass_completed"],
        "passes_received": counts["pass_received"],
        "balls_won": counts["ball_won"],
        "shots": counts["shot"],
        "ball_seen_share": round(ball.seen_fraction(), 3),
        "team_possession_share": _team_share(all_spells, team, target_team, fps),
    }
    events.sort(key=lambda e: e.t)
    return events, stats


def _team_share(all_spells: list[Spell], team, target_team: int, fps: float) -> float | None:
    """Share of the time either team had the ball that the followed player's team had it."""
    ours = theirs = 0
    for spell in all_spells:
        length = spell.end - spell.start + 1
        side = team(spell.owner)
        if side == target_team:
            ours += length
        elif side != OTHER:
            theirs += length
    return round(ours / (ours + theirs), 3) if ours + theirs else None
