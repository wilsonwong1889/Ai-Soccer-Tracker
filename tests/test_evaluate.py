import json
import subprocess

import pytest
from conftest import FixedDetector, needs_ffmpeg

from soccer_recorder.detect import Box
from soccer_recorder.evaluate import (
    LabelError,
    Labelled,
    average_precision,
    detect_images,
    iou,
    load_coco,
    match,
    score,
)


def person(x1, y1, x2, y2, s=0.9):
    return Box(x1, y1, x2, y2, s, "person")


def test_iou():
    a = person(0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, person(20, 20, 30, 30)) == 0.0
    assert iou(a, person(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_best_score_first_and_one_detection_per_target():
    target = [Labelled(0, 0, 10, 10, "player")]
    scored, found = match([person(0, 0, 10, 10, 0.6), person(1, 0, 10, 10, 0.9)], target, [])
    assert scored == [(0.9, True), (0.6, False)]  # the duplicate is a false positive
    assert found == [True]


def test_match_poor_overlap_is_a_miss_and_a_false_positive():
    scored, found = match([person(6, 0, 16, 10)], [Labelled(0, 0, 10, 10, "player")], [])
    assert scored == [(0.9, False)]
    assert found == [False]


def test_match_skips_detections_on_ignore_regions():
    bench = [Labelled(100, 0, 140, 40, "spectator", ignore=True)]
    scored, _ = match([person(100, 0, 140, 40)], [], bench)
    assert scored == []


def test_average_precision():
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    # Precision 1.0 up to recall 0.5, then 2/3 at recall 1.0.
    assert average_precision([(0.9, True), (0.8, False), (0.7, True)], 2) == pytest.approx(
        0.5 * 1.0 + 0.5 * 2 / 3
    )
    assert average_precision([], 3) == 0.0
    assert average_precision([(0.5, False)], 0) is None


def test_score_counts_per_class_with_threshold_and_heights():
    labels = {
        "a.jpg": [
            Labelled(0, 0, 10, 20, "player"),  # 20 px tall
            Labelled(50, 0, 60, 100, "Referee"),  # counts as a person target
            Labelled(200, 0, 240, 40, "spectator", ignore=True),
            Labelled(70, 70, 74, 74, "ball"),
        ],
        "b.jpg": [Labelled(0, 0, 30, 60, "goalkeeper")],
    }
    detections = {
        "a.jpg": [
            person(0, 0, 10, 20, 0.9),
            person(50, 0, 60, 100, 0.2),  # found, but below the threshold
            person(200, 0, 240, 40, 0.8),  # on a spectator: ignored
            person(300, 300, 310, 320, 0.7),  # nothing there: false positive
            Box(70, 70, 74, 74, 0.6, "sports ball"),
        ],
        "b.jpg": [],
    }
    report = score(detections, labels, threshold=0.3)
    people = report["classes"]["person"]
    assert people["labelled"] == 3
    assert people["true_positives"] == 1
    assert people["false_positives"] == 1
    assert people["missed"] == 2
    assert people["precision"] == 0.5
    assert people["recall"] == pytest.approx(1 / 3, abs=1e-4)
    # AP uses every detection, including the one under the threshold.
    assert people["ap50"] == pytest.approx(1 / 3 + (1 / 3) * (2 / 3), abs=1e-4)
    heights = people["recall_by_height_px"]
    assert heights["0-24"] == {"found": 1, "labelled": 1, "recall": 1.0}
    assert heights["48-96"]["labelled"] == 1 and heights["96+"]["labelled"] == 1
    assert report["classes"]["sports ball"]["recall"] == 1.0
    assert report["images"] == 2


def write_coco(path, file_name="frames/a.jpg"):
    path.write_text(json.dumps({
        "images": [{"id": 7, "file_name": file_name, "width": 320, "height": 180}],
        "categories": [{"id": 1, "name": "player"}, {"id": 2, "name": "Spectator"},
                       {"id": 3, "name": "ball"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 1, "bbox": [20, 20, 40, 120]},
            {"id": 2, "image_id": 7, "category_id": 2, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 7, "category_id": 1, "bbox": [5, 5, 5, 5], "iscrowd": 1},
            {"id": 4, "image_id": 7, "category_id": 3, "bbox": [120, 40, 8, 8]},
        ],
    }))
    return path


def test_load_coco(tmp_path):
    labels = load_coco(write_coco(tmp_path / "coco.json"))
    boxes = labels["a.jpg"]  # folder dropped from the CVAT file name
    assert boxes[0] == Labelled(20, 20, 60, 140, "player")
    assert boxes[1].ignore and boxes[1].category == "spectator"
    assert boxes[2].ignore  # iscrowd
    assert boxes[3].category == "ball"


@pytest.mark.parametrize("content", ["not json", "{}", '{"images": [], "categories": [{}]}'])
def test_load_coco_rejects_bad_files(tmp_path, content):
    path = tmp_path / "coco.json"
    path.write_text(content)
    with pytest.raises(LabelError):
        load_coco(path)


@needs_ffmpeg
def test_detect_images_maps_boxes_to_original_pixels(tmp_path):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=320x180",
         "-frames:v", "1", str(tmp_path / "a.jpg")],
        check=True,
    )
    labels = load_coco(write_coco(tmp_path / "coco.json"))
    detector = FixedDetector([Box(10, 10, 30, 70, 0.9, "person")])
    detections = detect_images(detector, labels, tmp_path, max_side=160)
    assert detector.shapes == [(90, 160, 3)]
    assert detections["a.jpg"] == [Box(20, 20, 60, 140, 0.9, "person")]
    report = score(detections, labels)
    assert report["classes"]["person"]["recall"] == 1.0


def test_detect_images_reports_missing_images(tmp_path):
    labels = load_coco(write_coco(tmp_path / "coco.json"))
    with pytest.raises(LabelError, match="not found"):
        detect_images(FixedDetector(), labels, tmp_path, max_side=None)
