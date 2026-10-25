import importlib.util
import json

import numpy as np
import pytest
from conftest import needs_ffmpeg
from synthetic import match_audio, match_scene, with_audio

from soccer_recorder.follow import Target, Trajectory
from soccer_recorder.highlights import pick_moments
from soccer_recorder.playercam import VERTICAL, CamParams, camera_path

has_pillow = importlib.util.find_spec("PIL") is not None
FPS = 10.0


def standing(box, n=50, lost_from=None):
    """A trajectory of a player standing still, optionally lost from a frame on."""
    times = np.arange(n) / FPS
    boxes = np.array([box] * n, float)
    state = ["seen"] * n
    if lost_from is not None:
        boxes[lost_from:] = np.nan
        state[lost_from:] = ["lost"] * (n - lost_from)
    return Trajectory(times=times, boxes=boxes, state=state)


# --- the virtual camera ---

def test_camera_zooms_so_the_player_fills_a_third_of_the_picture():
    trajectory = standing([940, 486, 980, 594])  # 108 px tall, in the middle of 1920x1080
    crops = camera_path(trajectory, trajectory.times, (1920, 1080))
    x1, y1, x2, y2 = crops[25]
    assert y2 - y1 == pytest.approx(108 / 0.33, rel=0.02)
    assert (x2 - x1) / (y2 - y1) == pytest.approx(16 / 9)
    assert (x1 + x2) / 2 == pytest.approx(960, abs=1) and (y1 + y2) / 2 == pytest.approx(540, abs=1)


def test_camera_stays_inside_the_frame_and_never_zooms_in_too_far():
    trajectory = standing([0, 1040, 10, 1060])  # tiny, in the bottom left corner
    crops = camera_path(trajectory, trajectory.times, (1920, 1080))
    assert (crops[:, 0] >= 0).all() and (crops[:, 1] >= 0).all()
    assert (crops[:, 2] <= 1920 + 1e-6).all() and (crops[:, 3] <= 1080 + 1e-6).all()
    assert np.allclose(crops[:, 3] - crops[:, 1], 0.25 * 1080)  # the closest zoom allowed


def test_camera_eases_out_to_the_whole_pitch_when_the_player_is_lost():
    trajectory = standing([940, 486, 980, 594], n=120, lost_from=60)
    crops = camera_path(trajectory, trajectory.times, (1920, 1080))
    heights = crops[:, 3] - crops[:, 1]
    assert heights[20] == pytest.approx(108 / 0.33, rel=0.02)
    assert heights[62] < heights[70] < heights[80]  # easing out, not jumping
    assert heights[-1] == pytest.approx(1080, rel=0.01)


def test_vertical_camera_has_the_phone_shape():
    trajectory = standing([940, 486, 980, 594])
    crops = camera_path(trajectory, trajectory.times, (1920, 1080), CamParams(out_size=VERTICAL))
    x1, y1, x2, y2 = crops[25]
    assert (x2 - x1) / (y2 - y1) == pytest.approx(9 / 16)


def test_camera_keeps_a_jittery_player_centred_and_pans_smoothly():
    from soccer_recorder.playercam import framing_report

    rng = np.random.default_rng(0)
    n = 300  # 30 s running across the pitch and back, with tracking noise and gaps
    times = np.arange(n) / FPS
    x = 960 + 700 * np.sin(times / 30 * 2 * np.pi)
    boxes = np.stack([x - 20, np.full(n, 480.0), x + 20, np.full(n, 588.0)], axis=1)
    boxes += rng.normal(0, 4, boxes.shape)
    state = ["seen"] * n
    for start in (50, 140, 220):  # hidden for half a second at a time
        boxes[start:start + 5] = np.nan
        state[start:start + 5] = ["lost"] * 5
    trajectory = Trajectory(times=times, boxes=boxes, state=state)
    for params in (CamParams(), CamParams(out_size=VERTICAL)):
        crops = camera_path(trajectory, times, (1920, 1080), params)
        report = framing_report(trajectory, times, crops, (1920, 1080))
        assert report["centred"] >= 0.95, report
        assert report["acceleration_max"] <= 1.0, report  # picture widths per second squared


# --- choosing highlight moments ---

def analytics(*events):
    return {"window": {"start_s": 0.0, "end_s": 600.0}, "events": [
        {"id": f"{kind}-{round(t * 10)}", "type": kind, "t": t, "end_t": None,
         "confidence": confidence} for kind, t, confidence in events]}


def test_moments_follow_the_reviews_and_the_confidence():
    found = analytics(("goal", 100.0, 0.95), ("shot", 200.0, 0.5), ("dribble", 300.0, 0.9),
                      ("pass_completed", 400.0, 0.3), ("touch", 500.0, 1.0))
    reviews = {"dribble-3000": "no", "pass_completed-4000": "yes"}
    moments = pick_moments(found, reviews)
    assert [m.event_ids for m in moments] == [["goal-1000"], ["pass_completed-4000"]]
    assert moments[0].caption == "GOAL  1:40"
    assert (moments[0].start, moments[0].end) == (94.0, 105.0)  # a lead-in, then the celebration


def test_overlapping_moments_merge_under_the_best_caption():
    moments = pick_moments(analytics(("dribble", 50.0, 0.9), ("shot", 53.0, 0.9),
                                     ("goal", 54.0, 0.95)), {})
    assert len(moments) == 1
    assert moments[0].caption.startswith("GOAL") and moments[0].start == 48.0
    assert set(moments[0].event_ids) == {"dribble-500", "shot-530", "goal-540"}


def test_the_reel_keeps_the_best_moments_when_too_long():
    found = analytics(*[("dribble", 20.0 * k, 0.7) for k in range(1, 20)], ("goal", 500.0, 0.95))
    moments = pick_moments(found, {}, max_total_s=20.0)
    assert sum(m.end - m.start for m in moments) <= 20.0
    assert any(m.caption.startswith("GOAL") for m in moments)
    assert [m.start for m in moments] == sorted(m.start for m in moments)


def test_moments_stay_inside_the_analysed_stretch():
    found = analytics(("goal", 2.0, 0.95))
    found["window"] = {"start_s": 0.0, "end_s": 4.0}
    (moment,) = pick_moments(found, {})
    assert (moment.start, moment.end) == (0.0, 4.0)


# --- rendering (needs FFmpeg and Pillow) ---

@pytest.fixture(scope="module")
def match_video(tmp_path_factory):
    root = tmp_path_factory.mktemp("cam")
    scene = match_scene()
    scene.duration = 4.0
    truth = scene.write(root / "silent.mp4")
    match_audio(root / "sound.wav", scene.duration, cheer=(2.0, 4.0), whistle=(3.0, 3.4))
    video = with_audio(root / "silent.mp4", root / "sound.wav", root / "match.mp4")
    boxes = np.array([b if b is not None else [np.nan] * 4 for b in truth["target"]], float)
    trajectory = Trajectory(times=scene.times, boxes=boxes,
                            state=["seen" if b is not None else "lost" for b in truth["target"]])
    return root, video, trajectory


@needs_ffmpeg
@pytest.mark.skipif(not has_pillow, reason="detect extra not installed")
@pytest.mark.parametrize("size", [(320, 180), (180, 320)])
def test_player_cam_video_has_the_size_and_the_sound(match_video, size):
    from soccer_recorder.media import probe
    from soccer_recorder.playercam import render_playercam

    root, video, trajectory = match_video
    out = root / f"cam-{size[0]}.mp4"
    written = render_playercam(video, trajectory, Target("10", "Emma"), out,
                               params=CamParams(out_size=size), caption="GOAL  0:03")
    info = probe(out)
    assert (info.display_width, info.display_height) == size
    assert info.has_audio and info.duration_s == pytest.approx(4.0, abs=0.2)
    assert written == pytest.approx(4.0 * 15, abs=2)


@needs_ffmpeg
@pytest.mark.skipif(not has_pillow, reason="detect extra not installed")
def test_highlight_reel_joins_the_moments(match_video):
    from soccer_recorder.highlights import Moment, render_reel
    from soccer_recorder.media import probe

    root, video, trajectory = match_video
    moments = [Moment(0.0, 1.0, "DRIBBLE  0:00", 1.0, ["a"]),
               Moment(2.5, 4.0, "GOAL  0:03", 1.0, ["b"])]
    seconds = render_reel(video, trajectory, Target("10", "Emma"), moments, root / "reel.mp4",
                          params=CamParams(out_size=(320, 180)))
    info = probe(root / "reel.mp4")
    assert seconds == pytest.approx(2.5)
    assert info.duration_s == pytest.approx(2.5, abs=0.3) and info.has_audio
    with pytest.raises(ValueError):
        render_reel(video, trajectory, Target("10"), [], root / "none.mp4")
    assert not (root / "none.mp4").exists()
    json.dumps([m.__dict__ for m in moments])  # moments are plain data for the website
