"""The highlight reel: the followed player's best moments, back to back.

Moments come from the analytics events:

- events the parent ticked ✓ always go in;
- events they crossed ✗ never do;
- unreviewed events go in if they're confident enough.

Each moment is a short clip around the event, with a little lead-in so
the play makes sense. Overlapping clips are merged. If the total is too
long, the best-scoring moments are kept (goals first), in game order. Every
clip is a player cam shot with a caption ("GOAL 12:31").
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .detect import Progress
from .media import probe, require_tool
from .playercam import DEFAULT_CAM, CamParams, render_playercam

# Event type: (seconds before, seconds after, weight, caption)
MOMENTS = {
    "goal": (6.0, 5.0, 1.0, "GOAL"),
    "possible_goal": (5.0, 4.0, 0.6, "BIG CHANCE"),
    "shot": (4.0, 3.0, 0.8, "SHOT"),
    "dribble": (2.0, 2.0, 0.6, "DRIBBLE"),
    "ball_won": (3.0, 2.0, 0.5, "WINS THE BALL"),
    "pass_completed": (3.0, 2.0, 0.4, "PASS"),
}
DEFAULT_MAX_TOTAL_S = 90.0
DEFAULT_MIN_CONFIDENCE = 0.6


@dataclass
class Moment:
    start: float
    end: float
    caption: str
    score: float
    event_ids: list[str]


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def pick_moments(analytics: dict, reviews: dict[str, str], *,
                 max_total_s: float = DEFAULT_MAX_TOTAL_S,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[Moment]:
    """Which stretches of the game go in the reel, in game order."""
    start_s, end_s = analytics["window"]["start_s"], analytics["window"]["end_s"]
    candidates = []
    for event in analytics["events"]:
        kind = event["type"]
        verdict = reviews.get(event["id"])
        if kind not in MOMENTS or verdict == "no":
            continue
        if verdict != "yes" and event["confidence"] < min_confidence:
            continue
        before, after, weight, caption = MOMENTS[kind]
        end = event["end_t"] if event["end_t"] is not None else event["t"]
        candidates.append(Moment(
            start=max(start_s, event["t"] - before), end=min(end_s, end + after),
            caption=f"{caption}  {_clock(event['t'])}",
            score=weight * event["confidence"] + (1.0 if verdict == "yes" else 0.0),
            event_ids=[event["id"]],
        ))
    candidates.sort(key=lambda m: m.start)
    merged: list[Moment] = []
    for moment in candidates:
        if merged and moment.start <= merged[-1].end:
            last = merged[-1]
            if moment.score > last.score:  # keep the better moment's caption
                last.caption = moment.caption
            last.end = max(last.end, moment.end)
            last.score = max(last.score, moment.score)
            last.event_ids += moment.event_ids
        else:
            merged.append(moment)
    chosen, total = [], 0.0
    for moment in sorted(merged, key=lambda m: m.score, reverse=True):
        length = moment.end - moment.start
        if total + length <= max_total_s:
            chosen.append(moment)
            total += length
    return sorted(chosen, key=lambda m: m.start)


def render_reel(video: str | Path, trajectory, target, moments: list[Moment],
                out_path: str | Path, *, params: CamParams = DEFAULT_CAM,
                on_progress: Callable[[Progress], None] | None = None) -> float:
    """Render each moment as a captioned player cam clip and join them. Returns seconds."""
    if not moments:
        raise ValueError("no moments to put in a reel yet: tick some on the Analytics page, "
                         "or analyse more of the game")
    info = probe(video)
    total = sum(m.end - m.start for m in moments)
    done = 0.0
    with tempfile.TemporaryDirectory() as temp:
        clips = []
        for k, moment in enumerate(moments):
            clip = Path(temp) / f"clip{k:03d}.mp4"

            def progress(p: Progress, offset=done) -> None:
                if on_progress is not None:
                    on_progress(Progress(frames=p.frames, video_s=offset + p.video_s,
                                         total_video_s=total, elapsed_s=p.elapsed_s))

            render_playercam(video, trajectory, target, clip, start_s=moment.start,
                             end_s=moment.end, params=params, caption=moment.caption,
                             info=info, on_progress=progress)
            clips.append(clip)
            done += moment.end - moment.start
        listing = Path(temp) / "clips.txt"
        listing.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips))
        # Every clip comes from the same encoder settings, so they join without re-encoding.
        result = subprocess.run(
            [require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"joining the clips failed: {result.stderr.strip()}")
    return total
