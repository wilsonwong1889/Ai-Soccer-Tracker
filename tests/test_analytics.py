import importlib.util
import json

import numpy as np
import pytest
from conftest import needs_ffmpeg
from synthetic import OracleDetector, match_audio, match_scene, with_audio

from soccer_recorder.analytics import (
    combine_goals,
    load_reviews,
    review_summary,
    save_goals,
    save_review,
)
from soccer_recorder.appearance import FEATURE_SIZE, PART_SIZE
from soccer_recorder.audio import Listening, Sound, background
from soccer_recorder.ball import ball_path
from soccer_recorder.events import TARGET, Event, EventParams, possession, spells
from soccer_recorder.follow import Trajectory
from soccer_recorder.teams import OTHER, split_teams
from soccer_recorder.tracking import Tracks

has_tracker = importlib.util.find_spec("trackers") is not None
FPS = 10.0


# --- ball path ---

def test_ball_path_follows_the_plausible_ball_and_fills_short_gaps():
    times = np.arange(30) / FPS
    frames, boxes, scores = [], [], []
    for i in range(30):
        if 10 <= i < 13 or i >= 20:
            continue  # hidden: a 0.3 s gap, then gone
        x = 100 + 5 * i
        candidates = [([x - 4, 196, x + 4, 204], 0.6)]
        if i == 5:  # a confident false ball far away (a head?)
            candidates.append(([600, 40, 608, 48], 0.9))
        if i == 6:  # something ball-coloured but player-sized
            candidates.append(([100, 100, 160, 160], 0.95))
        for box, score in candidates:
            frames.append(i)
            boxes.append(box)
            scores.append(score)
    path = ball_path(times, FPS, np.array(frames), np.array(boxes, float), np.array(scores), 60)
    assert path.xy[5][0] == pytest.approx(125)  # not the far-away one
    assert path.xy[6][0] == pytest.approx(130)  # not the oversized one
    assert path.state[11] == "filled" and path.xy[11][0] == pytest.approx(155)
    assert path.state[25] == "unknown"


# --- teams ---

def shirt_feature(bin_index: int) -> np.ndarray:
    f = np.zeros(FEATURE_SIZE, dtype=np.float32)
    f[[0, PART_SIZE + bin_index, 2 * PART_SIZE]] = 1.0
    return f


def test_split_teams_by_shirt_with_an_odd_one_out():
    ids, features = [], []
    for track, colour in [(1, 2), (2, 2), (3, 2), (4, 14), (5, 14), (6, 14), (7, 26)]:
        for _ in range(8):
            ids.append(track)
            features.append(shirt_feature(colour))
    teams = split_teams(np.array(ids), np.array(features))
    groups = {teams.of_track[t] for t in (1, 2, 3)}, {teams.of_track[t] for t in (4, 5, 6)}
    assert len(groups[0]) == 1 and len(groups[1]) == 1 and groups[0] != groups[1]
    assert teams.of_track[7] == OTHER or teams.of_track[7] in groups[0] | groups[1]
    assert teams.team_of(shirt_feature(2)[PART_SIZE:2 * PART_SIZE]) == teams.of_track[1]


# --- possession ---

def tiny_tracks(boxes_by_frame: dict[int, list[tuple[int, list[float]]]], n: int) -> Tracks:
    frame, track, boxes = [], [], []
    for i, items in boxes_by_frame.items():
        for track_id, box in items:
            frame.append(i)
            track.append(track_id)
            boxes.append(box)
    return Tracks(fps=FPS, start_s=0.0, times=np.arange(n) / FPS, frame=np.array(frame),
                  track=np.array(track), boxes=np.array(boxes, float).reshape(-1, 4),
                  scores=np.ones(len(frame)), features=np.zeros((len(frame), FEATURE_SIZE)),
                  size=(640, 360))


def test_possession_goes_to_the_nearest_feet_not_to_a_ball_at_chest_height():
    n = 3
    target = [100, 100, 120, 160]
    other = [200, 100, 220, 160]
    tracks = tiny_tracks({i: [(7, other)] for i in range(n)}, n)
    trajectory = Trajectory(times=tracks.times, boxes=np.array([target] * n, float),
                            state=["seen"] * n)
    ball = ball_path(tracks.times, FPS, np.arange(n),
                     np.array([[110, 154, 116, 160],  # at the target's feet
                               [206, 154, 212, 160],  # at the other player's feet
                               [108, 112, 114, 118]], float),  # chest height
                     np.full(n, 0.8), 60)
    assert possession(tracks, trajectory, ball, EventParams()) == [TARGET, 7, None]


def test_spells_bridge_short_losses_and_drop_blips():
    owners = [TARGET] * 5 + [None] * 3 + [TARGET] * 5 + [None] * 10 + [4] + [None] * 3
    found = spells(owners, FPS, EventParams())
    assert [(s.owner, s.start, s.end) for s in found] == [(TARGET, 0, 12)]  # 4 was a blip


# --- goals from evidence ---

def test_goal_confidence_grows_with_evidence():
    in_goal = Event("ball_in_goal", 30.0, 32.0, 0.55, {"goal": 1})
    cheer = Sound(31.0, 34.0, 20.0)
    whistle = Sound(40.0, 40.5, 0.8, 3200)
    alone = combine_goals([in_goal], Listening(has_audio=True))
    assert alone[0].type == "goal" and alone[0].confidence == pytest.approx(0.55)
    full = combine_goals([in_goal], Listening(True, [cheer], [whistle]))
    assert full[0].confidence == pytest.approx(0.95)
    assert full[0].details["evidence"] == ["ball in the goal", "cheering", "whistle"]


def test_cheering_alone_is_only_a_possible_goal_and_short_cheers_nothing():
    loud = Sound(100.0, 103.0, 22.0)
    short = Sound(200.0, 200.8, 22.0)
    quiet = Sound(300.0, 304.0, 9.0)
    found = combine_goals([], Listening(True, [loud, short, quiet], []))
    assert [e.type for e in found] == ["possible_goal"]
    assert 0.25 <= found[0].confidence <= 0.5


def test_background_is_not_raised_by_a_cheer_at_the_very_end():
    frame_s = 0.032
    db = np.full(int(10 / frame_s), -40.0)
    db[int(7 / frame_s):] = -18.0  # the last 3 s are a cheer
    assert background(db, frame_s)[-1] < -35


# --- reviews and goals files ---

def write_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "analytics.json").write_text(json.dumps({"events": [
        {"id": "goal-69", "type": "goal"}, {"id": "touch-1", "type": "touch"},
        {"id": "touch-45", "type": "touch"}]}))
    return run


def test_reviews_are_saved_cleared_and_summarised(tmp_path):
    run = write_run(tmp_path)
    save_review(run, "goal-69", "yes")
    save_review(run, "touch-1", "no")
    save_review(run, "touch-45", "yes")
    save_review(run, "touch-45", None)
    assert load_reviews(run) == {"goal-69": "yes", "touch-1": "no"}
    summary = review_summary(json.loads((run / "analytics.json").read_text()), load_reviews(run))
    assert summary["goal"] == {"total": 1, "reviewed": 1, "confirmed": 1}
    assert summary["touch"] == {"total": 2, "reviewed": 1, "confirmed": 0}
    with pytest.raises(LookupError):
        save_review(run, "pass-3", "yes")
    with pytest.raises(ValueError):
        save_review(run, "goal-69", "maybe")


@pytest.mark.parametrize("goals", [
    [[0.1, 0.1, 0.2, 0.2]] * 3,  # three goals
    [[0.1, 0.1, 1.2, 0.2]],  # off the frame
    [[0.1, 0.1, 0.1, 0.2]],  # a line, not a box
    [[0.1, 0.1, 0.2]],
])
def test_bad_goals_are_refused(tmp_path, goals):
    with pytest.raises(ValueError):
        save_goals(tmp_path, goals)


# --- the whole thing on a synthetic match ---

@pytest.fixture(scope="module")
def match(tmp_path_factory):
    if not has_tracker:
        pytest.skip("detect extra not installed")
    from soccer_recorder.follow import Seed, Target, follow_player
    from soccer_recorder.tracking import run_tracking

    root = tmp_path_factory.mktemp("match")
    scene = match_scene()
    truth = scene.write(root / "silent.mp4")
    match_audio(root / "sound.wav", scene.duration)
    video = with_audio(root / "silent.mp4", root / "sound.wav", root / "match.mp4")
    tracks = run_tracking(video, OracleDetector(truth, jitter=2, miss_rate=0.1), root / "t",
                          fps=15)
    x1, y1, x2, y2 = scene.players[0].box(0.5)
    follow_player(video, tracks, Seed(0.5, (x1 + x2) / 2, (y1 + y2) / 2),
                  Target("10", "Emma"), root / "run", render=False)
    width, height = scene.size
    goals = [(g[0], g[1], g[2], g[3]) for g in scene.goals]
    return root, video, tracks, goals, (width, height), truth


def first(events, kind):
    return next((e for e in events if e["type"] == kind), None)


@needs_ffmpeg
def test_match_events_are_found_where_they_happen(match):
    from soccer_recorder.analytics import analyse

    root, video, tracks, goals, _, _ = match
    result = analyse(video, tracks, root / "run", goals=goals)
    events = result["events"]
    expected = [  # (type, scripted time)
        ("dribble", 0.0), ("pass_completed", 2.0), ("pass_received", 4.3),
        ("shot", 6.5), ("ball_in_goal", 6.9), ("goal", 6.9), ("cheer", 7.2),
        ("whistle", 8.5),
    ]
    for kind, when in expected:
        event = first(events, kind)
        assert event is not None, f"no {kind}"
        assert event["t"] == pytest.approx(when, abs=0.5), kind
    goal = first(events, "goal")
    assert goal["confidence"] >= 0.9 and goal["details"]["by_followed_player"]
    assert not any(e["type"] in ("pass_lost", "ball_won", "possible_goal") for e in events)
    stats = result["stats"]
    assert (stats["dribbles"], stats["passes_completed"], stats["passes_received"],
            stats["shots"], stats["goals_in_game"]) == (2, 1, 1, 1, 1)
    assert stats["top_speed_kmh"] == pytest.approx(8.4, abs=1.0)  # 100 px/s, 60 px = 1.4 m
    assert result["teams"]["tracks"] == {"0": 2, "1": 2, "other": 0}
    assert sum(map(sum, result["stats"]["heatmap"]["values"])) > 0
    saved = json.loads((root / "run" / "analytics.json").read_text())
    assert saved["events"] == events


@needs_ffmpeg
def test_without_marked_goals_cheering_is_a_possible_goal(match):
    from soccer_recorder.analytics import analyse

    root, video, tracks, _, _, _ = match
    result = analyse(video, tracks, root / "run", goals=[])
    kinds = {e["type"] for e in result["events"]}
    assert "goal" not in kinds and "ball_in_goal" not in kinds and "shot" not in kinds
    assert "possible_goal" in kinds
    assert any("Mark the goals" in note for note in result["notes"])


@needs_ffmpeg
def test_without_sound_there_are_no_cheers(match, tmp_path):
    from soccer_recorder.analytics import analyse

    root, _, tracks, goals, _, _ = match
    result = analyse(root / "silent.mp4", tracks, root / "run", goals=goals)
    assert result["audio"]["has_audio"] is False
    goal = first(result["events"], "goal")
    assert goal["confidence"] == pytest.approx(0.55)  # the ball in the net, nothing else
    assert any("no sound" in note for note in result["notes"])
