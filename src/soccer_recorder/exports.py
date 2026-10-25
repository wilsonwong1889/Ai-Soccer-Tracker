"""Videos to keep and share, made from a follow run (R6).

- Player cam: the whole analysed stretch, cropped around the player.
- Highlight reel: their best moments, back to back (see highlights.py).

Each comes landscape (for a TV or laptop) or vertical (for a phone). Files
are written beside the run's other outputs, under fixed names, so making
one again replaces it. A half-written file is never left under the final
name. exports.json records what each file was made from, so the website can
say when a reel is out of date (for example after ticks and crosses).

Corrections also live here: re-following with an extra tap writes a new
run beside the old one, carrying the ticks and crosses over.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .analytics import analyse, load_goals, load_reviews, load_run_tracks, load_trajectory
from .detect import Progress
from .follow import Seed, Target, follow_player
from .highlights import Moment, pick_moments, render_reel
from .media import probe
from .playercam import (
    LANDSCAPE,
    VERTICAL,
    CamParams,
    camera_path,
    framing_report,
    render_playercam,
)

KINDS = ("playercam", "highlights")
EXPORTS_FILE = "exports.json"


def export_name(kind: str, vertical: bool = False) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    return f"{kind}{'-vertical' if vertical else ''}.mp4"


def _records(run_dir: Path) -> dict:
    path = Path(run_dir) / EXPORTS_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def reel_moments(run_dir: Path) -> list[Moment]:
    """The moments a reel made now would contain."""
    path = Path(run_dir) / "analytics.json"
    if not path.is_file():
        raise LookupError("this run has no analytics yet: press Re-analyse")
    return pick_moments(json.loads(path.read_text()), load_reviews(run_dir))


def _plan(moments: list[Moment]) -> list[list]:
    return [[round(m.start, 2), round(m.end, 2), m.caption] for m in moments]


def make_export(video: str | Path, run_dir: str | Path, kind: str, *, vertical: bool = False,
                on_progress: Callable[[Progress], None] | None = None) -> dict:
    """Render a player cam or highlight reel for a follow run. Returns what was made."""
    run_dir = Path(run_dir)
    name = export_name(kind, vertical)
    trajectory, saved = load_trajectory(run_dir / "target.json")
    target = Target(**saved["target"])
    params = CamParams(out_size=VERTICAL if vertical else LANDSCAPE)
    moments = reel_moments(run_dir) if kind == "highlights" else None
    if moments == []:
        raise ValueError("no moments for a reel yet: tick ✓ the ones you want, "
                         "or analyse more of the game")
    partial = run_dir / f".{Path(name).stem}-partial.mp4"  # hidden from the website
    started = time.perf_counter()
    try:
        if moments is None:
            render_playercam(video, trajectory, target, partial, params=params,
                             on_progress=on_progress)
        else:
            render_reel(video, trajectory, target, moments, partial, params=params,
                        on_progress=on_progress)
        info = probe(partial)
        seconds = info.duration_s
        partial.replace(run_dir / name)
    finally:
        partial.unlink(missing_ok=True)
    wall_s = time.perf_counter() - started
    source = probe(video)
    rate = min(source.fps, 30.0)
    times = np.arange(trajectory.times[0], trajectory.times[-1], 1.0 / rate)
    size = (source.display_width, source.display_height)
    framing = framing_report(trajectory, times, camera_path(trajectory, times, size, params), size)
    records = _records(run_dir)
    records[name] = {
        "kind": kind, "vertical": vertical, "seconds": round(seconds, 1),
        "size": [info.display_width, info.display_height],
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_s": round(wall_s, 1), "realtime_factor": round(seconds / wall_s, 2),
        "framing": framing,
        **({"moments": _plan(moments)} if moments is not None else {}),
    }
    (run_dir / EXPORTS_FILE).write_text(json.dumps(records, indent=2) + "\n")
    return {"file": name, **records[name]}


def list_exports(run_dir: Path) -> list[dict]:
    """The export files a run has, each with whether it's out of date."""
    run_dir = Path(run_dir)
    records = _records(run_dir)
    current = None
    if any((run_dir / export_name("highlights", v)).is_file() for v in (False, True)):
        try:  # only worth working out when there's a reel to compare with
            current = _plan(reel_moments(run_dir))
        except LookupError:
            pass
    found = []
    for kind in KINDS:
        for vertical in (False, True):
            name = export_name(kind, vertical)
            path = run_dir / name
            if not path.is_file():
                continue
            record = records.get(name, {})
            stale = (kind == "highlights" and current is not None
                     and record.get("moments") is not None and record["moments"] != current)
            found.append({"kind": kind, "vertical": vertical, "file": name,
                          "seconds": record.get("seconds"), "size_bytes": path.stat().st_size,
                          "out_of_date": stale})
    return found


def correct_run(video: str | Path, run_dir: str | Path, tap: Seed, out_dir: str | Path, *,
                goals_dir: Path | None = None, render: bool = True,
                on_progress: Callable[[Progress], None] | None = None) -> dict:
    """Follow again with one more tap, into a new run folder. Returns the follow metrics.

    The earlier taps are kept. Ticks and crosses carry over to moments that are
    still found.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    saved = json.loads((run_dir / "target.json").read_text())
    tracks = load_run_tracks(run_dir)
    if not tracks.times[0] - 0.5 <= tap.t <= tracks.times[-1] + 0.5:
        raise ValueError(f"tap a moment between {tracks.times[0]:.0f} and "
                         f"{tracks.times[-1]:.0f} s, the part that was analysed")
    seeds = [Seed(**s) for s in saved.get("seeds") or [saved["seed"]]]
    target = Target(**saved["target"])
    metrics = follow_player(video, tracks, seeds + [tap], target, out_dir, render=render,
                            on_progress=on_progress, tracks_run=saved.get("tracks_run"))
    previous = run_dir / "analytics.json"
    height = None
    if previous.is_file():
        height = json.loads(previous.read_text()).get("settings", {}).get("player_height_m")
    result = analyse(video, tracks, out_dir,
                     goals=load_goals(goals_dir, tracks.size) if goals_dir else [],
                     player_height_m=height)
    ids = {e["id"] for e in result["events"]}
    kept = {k: v for k, v in load_reviews(run_dir).items() if k in ids}
    if kept:
        (out_dir / "reviews.json").write_text(json.dumps(kept, indent=2) + "\n")
    return {**metrics, "taps": [asdict(s) for s in seeds + [tap]], "reviews_kept": len(kept)}
