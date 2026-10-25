import importlib.util

import numpy as np
import pytest
from conftest import needs_ffmpeg
from synthetic import OracleDetector, blocker_scene

pytestmark = [
    needs_ffmpeg,
    pytest.mark.skipif(importlib.util.find_spec("trackers") is None,
                       reason="detect extra not installed"),
]


@pytest.fixture(scope="module")
def scene_video(tmp_path_factory):
    scene = blocker_scene()
    scene.duration = 4.0
    path = tmp_path_factory.mktemp("scene") / "scene.mp4"
    truth = scene.write(path)
    return path, truth, scene


def test_run_tracking_saves_tracks_and_mot(tmp_path, scene_video):
    from soccer_recorder.tracking import Tracks, run_tracking, tracklets

    path, truth, scene = scene_video
    tracks = run_tracking(path, OracleDetector(truth), tmp_path, fps=scene.fps)
    assert len(tracks.times) == pytest.approx(scene.duration * scene.fps, abs=1)
    assert tracks.size == scene.size
    # Four players; the target and teammate vanish behind the block, so their
    # tracks end and new ones start when they come out.
    ids = {t.id for t in tracklets(tracks)}
    assert 4 <= len(ids) <= 6
    loaded = Tracks.load(tmp_path / "tracks.npz")
    assert np.array_equal(loaded.track, tracks.track)
    assert np.allclose(loaded.boxes, tracks.boxes)
    first = (tmp_path / "tracks.txt").read_text().splitlines()[0].split(",")
    assert len(first) == 10 and int(first[0]) >= 1 and int(first[1]) >= 1
    assert (tmp_path / "metrics.json").is_file()


def test_no_detections_gives_empty_tracks(tmp_path, scene_video):
    from soccer_recorder.follow import Seed, SeedNotFound, follow
    from soccer_recorder.tracking import run_tracking

    class Nothing:
        name = "nothing"

        def detect(self, image):
            return []

        def describe(self):
            return {"name": self.name}

    path, _, scene = scene_video
    tracks = run_tracking(path, Nothing(), tmp_path, fps=scene.fps)
    assert len(tracks.track) == 0 and len(tracks.times) > 0
    with pytest.raises(SeedNotFound):
        follow(tracks, Seed(1.0, 100, 100))


def test_load_or_track_reuses_matching_runs(tmp_path, scene_video):
    from soccer_recorder.tracking import load_or_track

    path, truth, scene = scene_video
    first = OracleDetector(truth)
    _, folder, reused = load_or_track(path, tmp_path, first, fps=scene.fps)
    assert not reused and first.calls > 0
    second = OracleDetector(truth)
    _, same_folder, reused = load_or_track(path, tmp_path, second, fps=scene.fps)
    assert reused and same_folder == folder and second.calls == 0
    _, other, reused = load_or_track(path, tmp_path, OracleDetector(truth), fps=scene.fps,
                                     duration_s=2)
    assert not reused and other != folder  # different window, different run
