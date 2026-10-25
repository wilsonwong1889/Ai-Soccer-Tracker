import importlib.util
import json

import numpy as np
import pytest
from conftest import needs_ffmpeg
from synthetic import OracleDetector, blocker_scene, crossing_scene, iou

from soccer_recorder.appearance import FEATURE_SIZE
from soccer_recorder.follow import (
    Params,
    Seed,
    SeedNotFound,
    Target,
    _smooth,
    build_trajectory,
    find_seed,
    follow,
    link_chain,
)
from soccer_recorder.tracking import Tracklet, split_on_appearance

has_tracker = importlib.util.find_spec("trackers") is not None
has_pillow = importlib.util.find_spec("PIL") is not None
FPS = 10.0


def feature(colour: int) -> np.ndarray:
    """A made-up appearance: all weight in one histogram bin per part."""
    f = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for part in range(3):
        f[part * 27 + colour] = 1.0
    return f


def tracklet(id_, start, n, x0, dx=5.0, colour=1, y=100.0, height=60.0):
    frames = np.arange(start, start + n)
    xs = x0 + dx * np.arange(n)
    boxes = np.stack([xs - 10, np.full(n, y - height), xs + 10, np.full(n, y)], axis=1)
    return Tracklet(id_, frames, boxes, np.full(n, 0.9), [feature(colour)] * n)


# --- smoothing and trajectories ---

def test_smooth_keeps_constants_and_reduces_noise():
    assert np.allclose(_smooth(np.full(20, 7.0), 2.0), 7.0)  # no sag at the ends
    rng = np.random.default_rng(0)
    noisy = np.linspace(0, 100, 200) + rng.normal(0, 3, 200)
    smoothed = _smooth(noisy, 3.0)
    assert np.std(np.diff(smoothed)) < np.std(np.diff(noisy)) / 3


def test_trajectory_fills_short_gaps_only():
    times = np.arange(100) / FPS
    a = tracklet(1, 0, 20, 100)
    b = tracklet(2, 40, 20, 300)  # 2 s after a ends: filled
    c = tracklet(3, 95, 5, 600)  # 3.5 s after b ends: not filled
    trajectory = build_trajectory(times, [a, b, c], FPS, Params(fill_gap_s=3.0))
    assert trajectory.state[:20] == ["seen"] * 20
    assert trajectory.state[20:40] == ["filled"] * 20
    assert trajectory.state[60:95] == ["lost"] * 35
    middle = trajectory.boxes[30]
    assert a.boxes[-1][0] < middle[0] < b.boxes[0][0]  # in between, on the straight line


def test_box_at_interpolates_between_frames_and_stops_when_lost():
    times = np.arange(30) / FPS
    trajectory = build_trajectory(times, [tracklet(1, 0, 10, 100, dx=10)], FPS,
                                  Params(smooth_center_s=0, smooth_size_s=0))
    box, state = trajectory.box_at(0.25)  # halfway between frames 2 and 3
    assert state == "seen"
    assert (box[0] + box[2]) / 2 == pytest.approx(125.0)
    assert trajectory.box_at(2.5) is None  # lost by then
    assert trajectory.box_at(-1) is None and trajectory.box_at(99) is None


# --- choosing the tapped player ---

def test_find_seed_prefers_the_box_tapped():
    times = np.arange(20) / FPS
    left, right = tracklet(1, 0, 20, 100, dx=0), tracklet(2, 0, 20, 140, dx=0)
    item, frame = find_seed([left, right], times, Seed(t=1.0, x=142, y=70))
    assert item is right and frame == 10


def test_find_seed_accepts_a_tap_just_beside_the_player():
    times = np.arange(20) / FPS
    item, _ = find_seed([tracklet(1, 0, 20, 100, dx=0)], times, Seed(t=1.0, x=118, y=70))
    assert item.id == 1


def test_find_seed_explains_misses():
    times = np.arange(20) / FPS
    with pytest.raises(SeedNotFound, match="No player was found"):
        find_seed([tracklet(1, 0, 20, 100, dx=0)], times, Seed(t=1.0, x=400, y=70))
    with pytest.raises(SeedNotFound, match="outside the part that was analysed"):
        find_seed([tracklet(1, 0, 20, 100, dx=0)], times, Seed(t=30.0, x=100, y=70))


# --- linking pieces of the same player ---

def test_links_the_piece_where_the_player_should_be():
    seed = tracklet(1, 0, 20, 100, dx=5)  # moving right, ends at x = 195
    ahead = tracklet(2, 30, 20, 245, dx=5)  # where it should be after 10 frames
    behind = tracklet(3, 30, 20, 60, dx=5)  # same colours, wrong place
    chain, links = link_chain([seed, ahead, behind], seed, 5, FPS, Params())
    assert [t.id for t in chain] == [1, 2]
    assert links[0].to_id == 2 and links[0].direction == "forward"


def test_links_the_piece_that_looks_like_the_player():
    seed = tracklet(1, 0, 20, 100, dx=5, colour=1)
    same_look = tracklet(2, 30, 20, 250, dx=5, colour=1)
    other_team = tracklet(3, 30, 20, 245, dx=5, colour=10)  # better position, wrong colours
    chain, _ = link_chain([seed, same_look, other_team], seed, 5, FPS, Params())
    assert [t.id for t in chain] == [1, 2]


def test_links_backwards_too_and_marks_close_calls_uncertain():
    seed = tracklet(1, 30, 20, 300, dx=5)
    before_a = tracklet(2, 0, 25, 170, dx=5)  # ends at x = 290, 5 frames before
    before_b = tracklet(3, 0, 25, 168, dx=5)  # practically the same
    chain, links = link_chain([seed, before_a, before_b], seed, 35, FPS, Params())
    assert len(chain) == 2 and links[0].direction == "backward"
    assert links[0].uncertain


def test_no_link_when_nothing_plausible():
    seed = tracklet(1, 0, 20, 100, dx=5)
    far_later = tracklet(2, 200, 20, 120)  # 18 s later: beyond max_gap_s
    chain, links = link_chain([seed, far_later], seed, 5, FPS, Params())
    assert [t.id for t in chain] == [1] and links == []


def test_split_where_colours_change():
    n = 30
    item = tracklet(1, 0, n, 100)
    item.features = [feature(1)] * 15 + [feature(10)] * 15  # an id swap onto an opponent
    pieces = split_on_appearance([item], FPS)
    assert [len(p.frames) for p in pieces] == [15, 15]
    assert pieces[0].id == 1 and pieces[1].id != 1
    steady = split_on_appearance([tracklet(2, 0, n, 100)], FPS)
    assert len(steady) == 1


# --- whole pipeline on synthetic games (needs the tracker) ---

def accuracy(scene, truth, who, trajectory):
    """(frames on the player, frames on someone else, visible frames)."""
    on = wrong = 0
    for i, gt in enumerate(truth[who]):
        if gt is None or trajectory.state[i] == "lost":
            continue
        if iou(trajectory.boxes[i], gt) >= 0.5:
            on += 1
        elif any(other[i] is not None and iou(trajectory.boxes[i], other[i]) >= 0.5
                 for name, other in truth.items() if name != who):
            wrong += 1
    return on, wrong, sum(g is not None for g in truth[who])


@needs_ffmpeg
@pytest.mark.skipif(not has_tracker, reason="detect extra not installed")
@pytest.mark.parametrize("make_scene, who", [
    (blocker_scene, "target"), (blocker_scene, "teammate"), (crossing_scene, "target"),
])
def test_follows_the_tapped_player_through_hiding_and_crossings(tmp_path, make_scene, who):
    from soccer_recorder.tracking import run_tracking

    scene = make_scene()
    truth = scene.write(tmp_path / "scene.mp4")
    detector = OracleDetector(truth, jitter=3, miss_rate=0.1)
    tracks = run_tracking(tmp_path / "scene.mp4", detector, tmp_path / "tracks", fps=scene.fps)
    player = next(p for p in scene.players if p.name == who)
    x1, y1, x2, y2 = player.box(1.0)
    trajectory = follow(tracks, Seed(1.0, (x1 + x2) / 2, (y1 + y2) / 2))
    on, wrong, visible = accuracy(scene, truth, who, trajectory)
    assert wrong == 0, "followed someone else"
    assert on >= 0.9 * visible


@needs_ffmpeg
@pytest.mark.skipif(not (has_tracker and has_pillow), reason="detect extra not installed")
def test_follow_player_writes_video_with_the_label(tmp_path):
    from soccer_recorder.follow import follow_player
    from soccer_recorder.frames import iter_frames
    from soccer_recorder.tracking import run_tracking

    scene = blocker_scene()
    scene.duration = 3.0
    truth = scene.write(tmp_path / "scene.mp4")
    tracks = run_tracking(tmp_path / "scene.mp4", OracleDetector(truth), tmp_path / "t",
                          fps=scene.fps)
    x1, y1, x2, y2 = scene.players[0].box(1.0)
    metrics = follow_player(tmp_path / "scene.mp4", tracks, Seed(1.0, (x1 + x2) / 2, y1 + 30),
                            Target(number="10", name="Emma"), tmp_path / "run")

    assert metrics["kind"] == "follow"
    assert metrics["target"] == {"number": "10", "name": "Emma"}
    assert metrics["coverage"]["seen"] > 0.8
    saved = json.loads((tmp_path / "run" / "target.json").read_text())
    assert saved["seed"]["t"] == 1.0 and len(saved["frames"]) == len(tracks.times)

    frame = next(f for f in iter_frames(tmp_path / "run" / "followed.mp4", max_side=None)
                 if f.time_s >= 1.0)
    # The amber tag sits above the player's head at t = 1 s.
    above = frame.image[max(0, y1 - 40):y1, x1 - 30:x2 + 30].reshape(-1, 3).astype(int)
    amber = (above[:, 2] > 200) & (above[:, 1] > 150) & (above[:, 0] < 80)
    assert amber.sum() > 50


def test_target_label():
    assert Target(number="7", name="Sam").label == "Sam  #7"
    assert Target(number="7").label == "#7"
