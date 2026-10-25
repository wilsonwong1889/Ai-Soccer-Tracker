"""Score a detector against hand-labelled frames (the R2 acceptance check).

Labels are COCO JSON, the format CVAT exports as "COCO 1.0": `images`
(id, file_name), `annotations` (image_id, category_id, bbox as
[x, y, width, height] in image pixels, optional iscrowd) and `categories`
(id, name). See docs/labeling.md for the label set.

A detection counts as found when it overlaps an unclaimed labelled box of a
matching category by IoU >= 0.5, best-scoring detections first. Detections
over an "ignore" region (spectators, the bench, anything off the playing
surface) count neither for nor against the detector.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .detect import BALL, PERSON, Box, Detector
from .frames import read_image

IOU_THRESHOLD = 0.5
# Detector label -> labelled category names it should find (case-insensitive).
DEFAULT_TARGETS = {
    PERSON: {"person", "player", "goalkeeper", "referee"},
    BALL: {"ball", "sports ball"},
}
IGNORE_CATEGORIES = {"ignore", "spectator", "crowd"}
# Labelled box heights in image pixels: recall per bucket shows whether
# small, far-away players are the ones being missed.
HEIGHT_BUCKETS = ((0, 24), (24, 48), (48, 96), (96, math.inf))


class LabelError(Exception):
    """The labels file can't be used."""


@dataclass(frozen=True)
class Labelled:
    x1: float
    y1: float
    x2: float
    y2: float
    category: str
    ignore: bool = False

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def load_coco(path: str | Path) -> dict[str, list[Labelled]]:
    """Labelled boxes per image file name (the base name, as CVAT exports it)."""
    try:
        data = json.loads(Path(path).read_text())
        categories = {c["id"]: c["name"].strip().lower() for c in data["categories"]}
        images = {image["id"]: Path(image["file_name"]).name for image in data["images"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise LabelError(f"{path}: not a COCO labels file ({error})") from error

    labels: dict[str, list[Labelled]] = {name: [] for name in images.values()}
    for annotation in data.get("annotations", []):
        try:
            x, y, w, h = (float(v) for v in annotation["bbox"])
            name = images[annotation["image_id"]]
            category = categories[annotation["category_id"]]
        except (KeyError, ValueError, TypeError) as error:
            raise LabelError(f"{path}: bad annotation {annotation.get('id')}: {error}") from error
        ignore = category in IGNORE_CATEGORIES or bool(annotation.get("iscrowd"))
        labels[name].append(Labelled(x, y, x + w, y + h, category, ignore))
    return labels


def iou(a, b) -> float:
    """Intersection over union of two boxes with x1, y1, x2, y2."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def match(
    detections: Iterable[Box],
    targets: list[Labelled],
    ignores: list[Labelled],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[tuple[float, bool]], list[bool]]:
    """Greedy matching, best score first.

    Returns ([(score, is_true_positive)] for counted detections, [found?] per
    target). Detections that miss every target but sit on an ignore region
    are left out entirely.
    """
    found = [False] * len(targets)
    scored: list[tuple[float, bool]] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        best, best_iou = None, iou_threshold
        for i, target in enumerate(targets):
            if not found[i]:
                overlap = iou(det, target)
                if overlap >= best_iou:
                    best, best_iou = i, overlap
        if best is not None:
            found[best] = True
            scored.append((det.score, True))
        elif any(iou(det, region) >= iou_threshold for region in ignores):
            continue
        else:
            scored.append((det.score, False))
    return scored, found


def average_precision(scored: list[tuple[float, bool]], n_targets: int) -> float | None:
    """Area under the precision-recall curve (all-point interpolation)."""
    if n_targets == 0:
        return None
    tp = fp = 0
    recalls, precisions = [], []
    for _, is_tp in sorted(scored, key=lambda item: item[0], reverse=True):
        tp += is_tp
        fp += not is_tp
        recalls.append(tp / n_targets)
        precisions.append(tp / (tp + fp))
    # Interpolate: precision at each recall is the best at that recall or beyond.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, previous = 0.0, 0.0
    for recall, precision in zip(recalls, precisions, strict=True):
        ap += (recall - previous) * precision
        previous = recall
    return ap


def _bucket_name(low: float, high: float) -> str:
    return f"{low}-{high}" if high != math.inf else f"{low}+"


def _bucket(height: float) -> str:
    height = max(0.0, height)
    for low, high in HEIGHT_BUCKETS:
        if low <= height < high:
            return _bucket_name(low, high)
    raise AssertionError("HEIGHT_BUCKETS must cover every height")


@dataclass
class ClassReport:
    targets: int = 0
    scored: list[tuple[float, bool]] = field(default_factory=list)
    # Height bucket -> [found, labelled].
    by_height: dict[str, list[int]] = field(
        default_factory=lambda: {_bucket_name(*bucket): [0, 0] for bucket in HEIGHT_BUCKETS}
    )

    def summary(self, threshold: float) -> dict:
        kept = [is_tp for score, is_tp in self.scored if score >= threshold]
        tp = sum(kept)
        fp = len(kept) - tp
        fn = self.targets - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / self.targets if self.targets else None
        f1 = None
        if precision is not None and recall is not None:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = average_precision(self.scored, self.targets)

        def r(value):
            return round(value, 4) if value is not None else None

        return {
            "labelled": self.targets,
            "true_positives": tp,
            "false_positives": fp,
            "missed": fn,
            "precision": r(precision),
            "recall": r(recall),
            "f1": r(f1),
            "ap50": r(ap),
            "recall_by_height_px": {
                bucket: {"found": found, "labelled": total,
                         "recall": r(found / total) if total else None}
                for bucket, (found, total) in self.by_height.items()
            },
        }


def score(
    detections: Mapping[str, list[Box]],
    labels: Mapping[str, list[Labelled]],
    *,
    targets: Mapping[str, set[str]] = DEFAULT_TARGETS,
    threshold: float = 0.3,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict:
    """Precision, recall, F1 and AP50 per detector label, over labelled images.

    `detections` must hold boxes in the labelled images' pixels, keyed like
    `labels`. Precision/recall/F1 count detections scoring >= `threshold`;
    AP50 uses every detection given.
    """
    reports = {label: ClassReport() for label in targets}
    for name, labelled in labels.items():
        ignores = [box for box in labelled if box.ignore]
        found_here = detections.get(name, [])
        for label, wanted in targets.items():
            wanted = {w.lower() for w in wanted}
            goals = [
                box for box in labelled if not box.ignore and box.category.lower() in wanted
            ]
            scored, found = match([d for d in found_here if d.label == label], goals, ignores,
                                  iou_threshold)
            report = reports[label]
            report.targets += len(goals)
            report.scored.extend(scored)
            for goal, hit in zip(goals, found, strict=True):
                bucket = report.by_height[_bucket(goal.height)]
                bucket[0] += hit
                bucket[1] += 1
    return {
        "images": len(labels),
        "iou_threshold": iou_threshold,
        "score_threshold": threshold,
        "classes": {label: report.summary(threshold) for label, report in reports.items()},
    }


def detect_images(
    detector: Detector,
    labels: Mapping[str, list[Labelled]],
    images_dir: str | Path,
    *,
    max_side: int | None,
    on_image: Callable[[int, int], None] | None = None,
) -> dict[str, list[Box]]:
    """Run `detector` on every labelled image, prepared like video frames.

    Boxes come back in the original images' pixels, ready for `score`.
    """
    images_dir = Path(images_dir)
    missing = [name for name in labels if not (images_dir / name).is_file()]
    if missing:
        raise LabelError(
            f"{len(missing)} labelled image(s) not found in {images_dir}, e.g. {missing[0]}"
        )
    results = {}
    for n, name in enumerate(sorted(labels), start=1):
        image, original_w, original_h = read_image(images_dir / name, max_side)
        height, width = image.shape[:2]
        sx, sy = original_w / width, original_h / height
        results[name] = [box.scaled(sx, sy) for box in detector.detect(image)]
        if on_image is not None:
            on_image(n, len(labels))
    return results
