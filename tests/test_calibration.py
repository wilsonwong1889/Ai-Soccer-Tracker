import json

import pytest
from conftest import needs_ffmpeg

from soccer_recorder.calibration import (
    ONLY_TICKED,
    load_calibration,
    review_stats,
    update_calibration,
)
from soccer_recorder.highlights import pick_moments
from soccer_recorder.library import ingest

pytestmark = needs_ffmpeg


def run_with(asset, stamp, events, reviews):
    run = asset.directory / "runs" / f"follow-n10-{stamp}"
    run.mkdir(parents=True)
    (run / "analytics.json").write_text(json.dumps({"events": [
        {"id": f"{kind}-{round(t * 10)}", "type": kind, "t": t, "end_t": None,
         "confidence": confidence} for kind, t, confidence in events]}))
    (run / "reviews.json").write_text(json.dumps(reviews))
    return run


@pytest.fixture
def library(tmp_path, landscape_clip):
    library = tmp_path / "lib"
    asset, _ = ingest(landscape_clip, library)
    # 12 dribbles: the confident ones (>= 0.75) were right, the rest wrong.
    dribbles = [("dribble", 10.0 * k, 0.5 + 0.04 * k) for k in range(12)]
    shots = [("shot", 300.0 + k, 0.9) for k in range(10)]  # always wrong
    passes = [("pass_completed", 400.0 + k, 0.7) for k in range(4)]  # too few to learn
    reviews = {f"{kind}-{round(t * 10)}": "yes" if (kind == "dribble" and c >= 0.75)
               or kind == "pass_completed" else "no" for kind, t, c in dribbles + shots + passes}
    run_with(asset, "20261025-100000", dribbles + shots + passes, reviews)
    return library, asset, reviews


def test_each_kind_learns_the_confidence_it_needs(library):
    library, _, _ = library
    stats = review_stats(library)
    assert stats["checked"] == 26 and stats["games"] == 1
    dribble = stats["types"]["dribble"]
    assert (dribble["checked"], dribble["right"]) == (12, 5)
    assert dribble["min_confidence"] == 0.75
    assert stats["types"]["shot"]["min_confidence"] == ONLY_TICKED
    assert stats["types"]["pass_completed"]["min_confidence"] is None  # 4 < 10 checked
    update_calibration(library)
    assert load_calibration(library) == {"dribble": 0.75, "shot": ONLY_TICKED}


def test_a_fixed_run_does_not_count_twice_and_its_verdicts_win(library):
    library, asset, reviews = library
    changed = {**reviews, "shot-3000": "yes"}
    run_with(asset, "20261025-110000", [("shot", 300.0, 0.9)], changed)
    stats = review_stats(library)
    assert stats["checked"] == 26
    assert stats["types"]["shot"]["right"] == 1


def test_the_reel_uses_what_was_learnt():
    analytics = {"window": {"start_s": 0.0, "end_s": 600.0}, "events": [
        {"id": "dribble-1000", "type": "dribble", "t": 100.0, "end_t": None, "confidence": 0.7},
        {"id": "shot-2000", "type": "shot", "t": 200.0, "end_t": None, "confidence": 0.95},
        {"id": "shot-3000", "type": "shot", "t": 300.0, "end_t": None, "confidence": 0.95},
    ]}
    assert len(pick_moments(analytics, {})) == 3
    learnt = {"dribble": 0.75, "shot": ONLY_TICKED}
    moments = pick_moments(analytics, {"shot-3000": "yes"}, learnt=learnt)
    assert [m.event_ids for m in moments] == [["shot-3000"]]  # ticked still goes in


def test_missing_or_broken_calibration_is_ignored(tmp_path):
    assert load_calibration(tmp_path) == {}
    (tmp_path / "calibration.json").write_text("{not json")
    assert load_calibration(tmp_path) == {}
