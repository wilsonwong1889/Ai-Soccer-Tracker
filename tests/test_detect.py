import json

import pytest
from conftest import DURATION, FPS, FixedDetector, needs_ffmpeg

from soccer_recorder.detect import Box, read_detections, run_detection
from soccer_recorder.media import probe

pytestmark = needs_ffmpeg


def test_run_detection_writes_boxes_in_source_pixels(tmp_path, landscape_clip):
    detector = FixedDetector()
    metrics = run_detection(landscape_clip, detector, tmp_path / "run", max_side=160)

    # The 320x180 source is analysed at 160x90, so boxes scale by 2.
    assert detector.shapes[0] == (90, 160, 3)
    frames = read_detections(tmp_path / "run" / "detections.jsonl")
    assert len(frames) == pytest.approx(DURATION * FPS, abs=1)
    first = frames[0]
    assert first["frame"] == 0 and first["t"] == 0.0
    labels = [box.label for box in first["boxes"]]
    assert labels == ["person", "person", "sports ball"]  # the chair is filtered out
    assert first["boxes"][0] == Box(20.0, 20.0, 60.0, 140.0, 0.9, "person")

    assert metrics["frames"] == len(frames)
    assert metrics["settings"]["detection_frame_size"] == [160, 90]
    assert metrics["detections"]["person"]["per_frame"] == 2.0
    assert metrics["detections"]["sports ball"]["frames_with_any"] == 1.0
    assert metrics["detections"]["person"]["mean_score"] == pytest.approx(0.65)
    assert metrics["person_height_px"]["p50"] == pytest.approx(92.0)  # 120 and 64 tall
    assert metrics["speed"]["realtime_factor"] > 0
    assert metrics["window"]["seconds"] == pytest.approx(DURATION, abs=0.1)
    assert metrics["detector"]["name"] == "fixed"
    assert json.loads((tmp_path / "run" / "metrics.json").read_text()) == metrics

    video = probe(tmp_path / "run" / "annotated.mp4")
    assert (video.display_width, video.display_height) == (160, 90)
    assert video.duration_s == pytest.approx(DURATION, abs=0.15)


def test_run_detection_window_and_rate(tmp_path, landscape_clip):
    metrics = run_detection(
        landscape_clip, FixedDetector(), tmp_path, fps=5, start_s=1.0, duration_s=0.6,
        write_video=False,
    )
    frames = read_detections(tmp_path / "detections.jsonl")
    assert len(frames) == pytest.approx(3, abs=1)
    assert frames[0]["t"] == pytest.approx(1.0)
    assert frames[1]["t"] == pytest.approx(1.2)
    assert metrics["window"] == {"start_s": 1.0, "seconds": 0.6}
    assert not (tmp_path / "annotated.mp4").exists()


def test_run_detection_on_rotated_video_uses_upright_size(tmp_path, portrait_clip):
    detector = FixedDetector()
    run_detection(portrait_clip, detector, tmp_path, max_side=160, write_video=False)
    assert detector.shapes[0] == (160, 90, 3)


def test_detector_failure_propagates_and_drops_partial_video(tmp_path, landscape_clip):
    with pytest.raises(RuntimeError, match="crashed"):
        run_detection(landscape_clip, FixedDetector(fail_after=3), tmp_path)
    assert not (tmp_path / "annotated.mp4").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_progress_is_reported(tmp_path, landscape_clip):
    seen = []
    run_detection(landscape_clip, FixedDetector(), tmp_path, write_video=False,
                  on_progress=seen.append)
    assert len(seen) == pytest.approx(DURATION * FPS, abs=1)
    assert seen[-1].total_video_s == pytest.approx(DURATION, abs=0.1)
    assert seen[-1].video_s == pytest.approx(DURATION - 1 / FPS, abs=0.1)
