import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from soccer_recorder.rfdetr_detector import (
    SIZES,
    DetectorUnavailable,
    RFDETRDetector,
    check_size,
    label_names,
    parse_shape,
)

has_rfdetr = importlib.util.find_spec("rfdetr") is not None


@pytest.mark.parametrize("size", ["nano", "small", "medium", "large"])
def test_apache_sizes_are_allowed(size):
    assert check_size(size) == size


@pytest.mark.parametrize("size", ["atto", "femto", "pico", "xlarge", "2xlarge", "base"])
def test_other_sizes_are_refused_with_the_license_reason(size):
    with pytest.raises(ValueError, match="Apache-2.0"):
        check_size(size)


def test_only_apache_sizes_are_offered():
    assert set(SIZES) == {"nano", "small", "medium", "large"}


def test_parse_shape():
    assert parse_shape("704x384") == (704, 384)
    assert parse_shape("704X384") == (704, 384)
    assert parse_shape(None) is None
    for bad in ("704", "axb", "0x384", "-1x2"):
        with pytest.raises(ValueError):
            parse_shape(bad)


def test_label_names_prefers_names_rfdetr_resolved():
    detections = SimpleNamespace(data={"class_name": np.array(["person", "sports ball"])})
    assert label_names(detections, [1, 37]) == ["person", "sports ball"]


def test_label_names_falls_back_to_coco_ids_not_list_positions():
    # Regression: COCO-pretrained RF-DETR emits COCO ids (person = 1), but
    # model.class_names is a 0-based list where position 1 is "bicycle".
    detections = SimpleNamespace(data={})
    coco = {1: "person", 37: "sports ball", 62: "chair"}
    assert label_names(detections, [1, 37, 62, 99], fallback=coco) == [
        "person", "sports ball", "chair", "99",
    ]


@pytest.mark.skipif(not has_rfdetr, reason="detect extra not installed")
def test_rfdetr_coco_ids_match_what_we_rely_on():
    from rfdetr.assets.coco_classes import COCO_CLASSES

    assert COCO_CLASSES[1] == "person"
    assert COCO_CLASSES[37] == "sports ball"


def test_missing_rfdetr_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", None)  # makes `import rfdetr` fail
    with pytest.raises(DetectorUnavailable, match="uv sync --extra detect"):
        RFDETRDetector("small", device="cpu")


@pytest.mark.skipif(
    not has_rfdetr or not os.environ.get("SOCCER_RECORDER_MODEL_TESTS"),
    reason="set SOCCER_RECORDER_MODEL_TESTS=1 to load real weights (downloads ~370 MB once)",
)
def test_real_model_runs_and_describes_itself():
    detector = RFDETRDetector("small", device="cpu", shape=(704, 384))
    assert detector.detect(np.zeros((360, 640, 3), dtype=np.uint8)) == []
    described = detector.describe()
    assert described["weights_license"] == "Apache-2.0"
    assert described["input_shape"] == [704, 384]
    with pytest.raises(ValueError, match="multiples of"):
        RFDETRDetector("small", device="cpu", shape=(700, 384))
