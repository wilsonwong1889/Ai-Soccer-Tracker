"""Learning from ticks and crosses: how often each kind of moment is right.

Every ✓ or ✗ on the website is saved in the run's reviews.json. This reads
them across every game in the library and works out, for each kind of
moment (dribble, shot, ...):

- how many were checked, and how many were right;
- how often they were right at or above each confidence;
- the confidence the highlight reel should require: the lowest at which at
  least 80% of the checked ones were right. Until 10 of a kind have been
  checked, the reel keeps its default (60%). If no confidence reaches 80%,
  that kind only goes in when ticked.

No model is trained: it's counting, so it's transparent and needs no
footage beyond your own library. The result is saved as calibration.json in
the library folder, and the highlight reel uses it.

A fixed run carries the ticks of the run it fixed, so each moment is counted
once per video (by its id), using the newest run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .library import list_assets

CALIBRATION_FILE = "calibration.json"
MIN_REVIEWED = 10
TARGET_PRECISION = 0.8
GRID = tuple(round(0.3 + 0.05 * k, 2) for k in range(14))  # 0.30 to 0.95
ONLY_TICKED = 1.01  # above any confidence: goes in the reel only when ticked


def _checked_moments(library: Path | None) -> list[dict]:
    """One entry per checked moment per video: type, confidence, right or not."""
    found: dict[tuple[str, str], dict] = {}
    for asset in list_assets(library):
        runs = sorted((asset.directory / "runs").glob("follow-*"), key=lambda p: p.name[-15:])
        for run in runs:  # oldest first, so the newest run's verdict wins
            reviews_path = run / "reviews.json"
            if not reviews_path.is_file() or not (run / "analytics.json").is_file():
                continue
            reviews = json.loads(reviews_path.read_text())
            if not reviews:
                continue
            events = {e["id"]: e for e in json.loads((run / "analytics.json").read_text())
                      .get("events", [])}
            for event_id, verdict in reviews.items():
                event = events.get(event_id)
                if event is None or verdict not in ("yes", "no"):
                    continue
                found[(asset.asset_id, event_id)] = {
                    "type": event["type"], "confidence": float(event["confidence"]),
                    "right": verdict == "yes", "asset_id": asset.asset_id,
                }
    return list(found.values())


def review_stats(library: Path | None = None) -> dict:
    """Per kind of moment: checked, right, precision, and the confidence the reel needs."""
    moments = _checked_moments(library)
    stats = {}
    for kind in sorted({m["type"] for m in moments}):
        mine = [m for m in moments if m["type"] == kind]
        right = sum(m["right"] for m in mine)
        levels = []
        for level in GRID:
            above = [m for m in mine if m["confidence"] >= level]
            if above:
                levels.append({"from": level, "checked": len(above),
                               "precision": round(sum(m["right"] for m in above) / len(above),
                                                  3)})
        threshold = None
        if len(mine) >= MIN_REVIEWED:
            threshold = next((lv["from"] for lv in levels if lv["precision"] >= TARGET_PRECISION
                              and lv["checked"] >= MIN_REVIEWED // 2), ONLY_TICKED)
        stats[kind] = {
            "checked": len(mine), "right": right,
            "precision": round(right / len(mine), 3),
            "games": len({m["asset_id"] for m in mine}),
            "by_confidence": levels,
            "min_confidence": threshold,  # None: not enough checked to learn yet
        }
    return {"checked": len(moments), "games": len({m["asset_id"] for m in moments}),
            "types": stats}


def update_calibration(library: Path) -> dict:
    """Re-learn from all the ticks and crosses, and save calibration.json."""
    stats = review_stats(library)
    calibration = {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "min_confidence": {kind: s["min_confidence"] for kind, s in stats["types"].items()
                           if s["min_confidence"] is not None},
        "stats": stats,
    }
    Path(library).mkdir(parents=True, exist_ok=True)
    temp = Path(library) / f".{CALIBRATION_FILE}.tmp"
    temp.write_text(json.dumps(calibration, indent=2) + "\n")
    temp.replace(Path(library) / CALIBRATION_FILE)
    return calibration


def load_calibration(library: Path) -> dict[str, float]:
    """The confidence each kind of moment needs for the reel, where it has been learnt."""
    path = Path(library) / CALIBRATION_FILE
    try:
        return {k: float(v) for k, v in json.loads(path.read_text())["min_confidence"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
