"""Analytics for a followed player: events, goals and statistics.

Runs after following, from saved data only (tracks.npz, target.json and the
video's sound), so it takes seconds and can be re-run, for example after the
parent marks where the goals are. Writes analytics.json next to the run.

Goals are combined from independent evidence, and the confidence says how much:

- the ball seen inside a marked goal: 0.55
- cheering starting up to 10 s after: +0.3
- a whistle up to 20 s after: +0.1
- cheering alone, loud and long, with no marked goal involved: a "possible
  goal" at 0.25 to 0.5. Near misses get cheered too.

None of this is validated on real games yet. Each event can be confirmed or
rejected on the website; reviews are kept in reviews.json and are how the
rules get measured (R7).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .audio import Listening, listen
from .ball import ball_path
from .events import Event, EventParams, detect_events
from .follow import Target, Trajectory
from .teams import OTHER, SHIRT, split_teams
from .tracking import Tracks

GOALS_FILE = "goals.json"  # per video: the goals the parent marked
REVIEWS_FILE = "reviews.json"  # per run: {event id: "yes" | "no"}


def load_trajectory(path: Path) -> tuple[Trajectory, dict]:
    """The followed player's path, as follow_player saved it in target.json."""
    data = json.loads(Path(path).read_text())
    frames = data["frames"]
    boxes = np.array([f["box"] if f["box"] else [np.nan] * 4 for f in frames], dtype=float)
    trajectory = Trajectory(times=np.array([f["t"] for f in frames], dtype=float),
                            boxes=boxes.reshape(-1, 4), state=[f["state"] for f in frames])
    return trajectory, data


def load_goals(asset_dir: Path, size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """Marked goals, converted from fractions of the frame to source pixels."""
    path = Path(asset_dir) / GOALS_FILE
    if not path.is_file():
        return []
    width, height = size
    goals = []
    for x1, y1, x2, y2 in json.loads(path.read_text()).get("goals", []):
        goals.append((min(x1, x2) * width, min(y1, y2) * height,
                      max(x1, x2) * width, max(y1, y2) * height))
    return goals


def save_goals(asset_dir: Path, goals: list[list[float]]) -> None:
    clean = []
    for goal in goals:
        if len(goal) != 4 or not all(0.0 <= float(v) <= 1.0 for v in goal):
            raise ValueError("each goal is four fractions of the frame: x1, y1, x2, y2")
        x1, y1, x2, y2 = (float(v) for v in goal)
        if abs(x2 - x1) < 0.01 or abs(y2 - y1) < 0.01:
            raise ValueError("a goal must be a box, not a point")
        clean.append([round(v, 4) for v in (x1, y1, x2, y2)])
    if len(clean) > 2:
        raise ValueError("mark at most two goals")
    (Path(asset_dir) / GOALS_FILE).write_text(json.dumps({"goals": clean}) + "\n")


def _target_shirt(tracks: Tracks, trajectory: Trajectory) -> np.ndarray | None:
    """The followed player's average shirt colours, from the tracked boxes that are them."""
    rows = []
    for row, frame in enumerate(tracks.frame):
        if trajectory.state[frame] == "lost":
            continue
        a, b = tracks.boxes[row], trajectory.boxes[frame]
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        if union > 0 and ix * iy / union > 0.5 and not np.isnan(tracks.features[row, 0]):
            rows.append(tracks.features[row, SHIRT])
    return np.mean(rows, axis=0) if rows else None


def combine_goals(events: list[Event], listening: Listening) -> list[Event]:
    """Goal candidates from the ball in a goal, cheering and whistles."""
    goals = []
    used_cheers = set()
    shots = [e for e in events if e.type == "shot"]
    for event in (e for e in events if e.type == "ball_in_goal"):
        confidence, evidence = 0.55, ["ball in the goal"]
        for k, cheer in enumerate(listening.cheers):
            if event.t - 1.0 <= cheer.start <= event.t + 10.0:
                confidence += 0.3
                evidence.append("cheering")
                used_cheers.add(k)
                break
        if any(event.t <= w.start <= event.t + 20.0 for w in listening.whistles):
            confidence += 0.1
            evidence.append("whistle")
        by_target = any(event.t - 3.0 <= s.end_t <= event.t + 0.5 for s in shots)
        goals.append(Event("goal", event.t, event.end_t, min(confidence, 0.95),
                           {"evidence": evidence, "goal": event.details.get("goal"),
                            "by_followed_player": by_target}))
    for k, cheer in enumerate(listening.cheers):
        if k in used_cheers or cheer.end - cheer.start < 1.5 or cheer.strength < 12.0:
            continue
        if any(abs(g.t - cheer.start) <= 10.0 for g in goals):
            continue
        whistle = any(cheer.start <= w.start <= cheer.start + 20.0 for w in listening.whistles)
        confidence = 0.25 + (0.1 if whistle else 0.0) + min(0.15, (cheer.strength - 12.0) / 40)
        evidence = ["cheering"] + (["whistle"] if whistle else [])
        goals.append(Event("possible_goal", cheer.start, cheer.end, confidence,
                           {"evidence": evidence, "loudness_db": round(cheer.strength, 1)}))
    return goals


def analyse(video: str | Path, tracks: Tracks, run_dir: str | Path, *,
            goals: list[tuple[float, float, float, float]] = (),
            player_height_m: float | None = None, audio: bool = True) -> dict:
    """Write analytics.json for a follow run folder (which holds target.json)."""
    run_dir = Path(run_dir)
    trajectory, saved = load_trajectory(run_dir / "target.json")
    target = Target(**saved["target"])
    params = EventParams(player_height_m=player_height_m or EventParams.player_height_m)
    heights = tracks.boxes[:, 3] - tracks.boxes[:, 1]
    ball = ball_path(tracks.times, tracks.fps, tracks.ball_frame, tracks.ball_boxes,
                     tracks.ball_scores, float(np.median(heights)) if len(heights) else 50.0)
    teams = split_teams(tracks.track, tracks.features)
    target_team = teams.team_of(_target_shirt(tracks, trajectory))
    events, stats = detect_events(tracks, trajectory, ball, teams, target_team, list(goals),
                                  params)
    start = float(tracks.times[0]) if len(tracks.times) else 0.0
    end = float(tracks.times[-1]) if len(tracks.times) else 0.0
    listening = (listen(video, start, end - start + 1.0 / tracks.fps) if audio
                 else Listening(has_audio=False))
    events += combine_goals(events, listening)
    events += [Event("cheer", c.start, c.end, min(0.9, 0.3 + c.strength / 40),
                     {"loudness_db": round(c.strength, 1)}) for c in listening.cheers]
    events += [Event("whistle", w.start, w.end, min(0.9, w.strength),
                     {"frequency_hz": round(w.frequency or 0)}) for w in listening.whistles]
    events.sort(key=lambda e: e.t)
    team_sizes = {"0": 0, "1": 0, "other": 0}
    for team in teams.of_track.values():
        team_sizes["other" if team == OTHER else str(team)] += 1
    width, height = tracks.size
    result = {
        "kind": "analytics",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "target": asdict(target),
        "window": {"start_s": round(start, 2), "end_s": round(end, 2)},
        "settings": {
            "player_height_m": params.player_height_m,
            "goals": [[round(g[0] / width, 4), round(g[1] / height, 4),
                       round(g[2] / width, 4), round(g[3] / height, 4)] for g in goals]
            if width and height else [],
            "audio": listening.has_audio,
        },
        "stats": {**stats,
                  "goals_in_game": sum(e.type == "goal" for e in events),
                  "possible_goals": sum(e.type == "possible_goal" for e in events)},
        "events": [e.to_dict() for e in events],
        "audio": {"has_audio": listening.has_audio, "step_s": listening.loudness_step_s,
                  "loudness_rise_db": listening.loudness},
        "ball": {"seen_share": round(ball.seen_fraction(), 3)},
        "teams": {"tracks": team_sizes, "followed_player_team": target_team},
        "notes": [
            f"Distances and speeds assume the player is {params.player_height_m:g} m tall "
            "and are estimates.",
            "Events are automatic guesses from the video and its sound. Tick or cross them "
            "to help measure how good they are.",
        ] + ([] if goals else ["Mark the goals to detect the ball going in."])
          + ([] if listening.has_audio else ["This video has no sound: no cheering or "
                                            "whistles."]),
    }
    (run_dir / "analytics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def analyse_run(video: str | Path, run_dir: Path, *, goals_dir: Path | None = None,
                player_height_m: float | None = None) -> dict:
    """Analyse a follow run folder, loading the tracks it was made from.

    Keeps the player height from an earlier analysis unless a new one is given.
    """
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "target.json").read_text())
    tracks_run = saved.get("tracks_run")
    if not tracks_run or not (run_dir.parent / tracks_run / "tracks.npz").is_file():
        raise LookupError("the tracking this run was made from is gone; follow the player again")
    tracks = Tracks.load(run_dir.parent / tracks_run / "tracks.npz")
    if player_height_m is None and (run_dir / "analytics.json").is_file():
        previous = json.loads((run_dir / "analytics.json").read_text())
        player_height_m = previous.get("settings", {}).get("player_height_m")
    goals = load_goals(goals_dir, tracks.size) if goals_dir else []
    return analyse(video, tracks, run_dir, goals=goals, player_height_m=player_height_m)


def load_reviews(run_dir: Path) -> dict[str, str]:
    path = Path(run_dir) / REVIEWS_FILE
    return json.loads(path.read_text()) if path.is_file() else {}


def save_review(run_dir: Path, event_id: str, verdict: str | None) -> dict[str, str]:
    """Record "yes" (it happened), "no" (it didn't) or None (clear) for one event."""
    if verdict not in ("yes", "no", None):
        raise ValueError("verdict must be yes, no or empty")
    analytics = json.loads((Path(run_dir) / "analytics.json").read_text())
    if event_id not in {e["id"] for e in analytics["events"]}:
        raise LookupError(f"no event {event_id!r} in this run")
    reviews = load_reviews(run_dir)
    if verdict is None:
        reviews.pop(event_id, None)
    else:
        reviews[event_id] = verdict
    (Path(run_dir) / REVIEWS_FILE).write_text(json.dumps(reviews, indent=2) + "\n")
    return reviews


def review_summary(analytics: dict, reviews: dict[str, str]) -> dict:
    """Per event type: how many were reviewed and how many were right."""
    summary: dict[str, dict[str, int]] = {}
    for event in analytics["events"]:
        verdict = reviews.get(event["id"])
        entry = summary.setdefault(event["type"], {"total": 0, "reviewed": 0, "confirmed": 0})
        entry["total"] += 1
        if verdict:
            entry["reviewed"] += 1
            entry["confirmed"] += verdict == "yes"
    return summary
