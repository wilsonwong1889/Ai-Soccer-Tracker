"""Fine-tune the detector on our own labelled footage (docs/training.md).

Needs the `train` extra (`uv sync --extra detect --extra train`). Starts from
RF-DETR's COCO-pretrained Apache-2.0 weights and trains only on a set built
by `soccer-recorder dataset`, so every frame has recorded training
permission. Nothing is uploaded: training runs on this computer, and RF-DETR
reads the dataset from disk.

A run folder holds RF-DETR's own outputs plus model.json, the model card:
what it started from, what it learnt from (the datasheet), and how it scored
on the held-out test frames before and after. A fine-tuned model is only
used when you pass `--weights <run folder>`, and only worth it if it scored
better.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .dataset import DATASHEET, load_datasheet
from .detect import BALL, PERSON
from .frames import DEFAULT_MAX_SIDE
from .rfdetr_detector import (
    DEFAULT_THRESHOLD,
    SIZES,
    WEIGHTS_LICENSE,
    DetectorUnavailable,
    check_size,
    pick_device,
)

MODEL_CARD = "model.json"
WEIGHTS = "rfdetr/checkpoint_best_total.pth"


class TrainingUnavailable(DetectorUnavailable):
    """RF-DETR's training extra isn't installed."""


def load_model_card(path: str | Path) -> tuple[dict, Path]:
    """The model card and weights file of a fine-tuned model (its run folder or .pth)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if not (folder / MODEL_CARD).is_file() and (folder.parent / MODEL_CARD).is_file():
        folder = folder.parent  # the .pth inside rfdetr/
    card_path = folder / MODEL_CARD
    if not card_path.is_file():
        raise ValueError(f"{path} isn't a model made by `soccer-recorder train-detector` (no "
                         f"{MODEL_CARD}). Only those can be used, so every model's training "
                         "data and licence are known")
    card = json.loads(card_path.read_text())
    weights = folder / card["weights"]
    if not weights.is_file():
        raise ValueError(f"{weights} is missing")
    check_size(card["size"])
    return card, weights


def _test_score(dataset_dir: Path, size: str, device: str, weights: Path | None,
                threshold: float, max_side: int | None) -> dict:
    from .evaluate import detect_images, load_coco, score
    from .rfdetr_detector import RFDETRDetector

    labels = load_coco(dataset_dir / "test" / "_annotations.coco.json")
    detector = RFDETRDetector(size, device=device, threshold=min(threshold, 0.05),
                              weights=weights)
    detections = detect_images(detector, labels, dataset_dir / "test", max_side=max_side)
    report = score(detections, labels, threshold=threshold)
    return {label: {k: report["classes"][label][k] for k in ("precision", "recall", "f1", "ap50")}
            for label in (PERSON, BALL)}


def _better(before: dict, after: dict) -> bool:
    """Better when players' AP50 improves and the ball's doesn't drop by more than 0.02."""
    def ap(report, label):
        return report[label]["ap50"] if report[label]["ap50"] is not None else 0.0

    return (ap(after, PERSON) > ap(before, PERSON)
            and ap(after, BALL) >= ap(before, BALL) - 0.02)


def train_detector(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    size: str = "small",
    epochs: int = 20,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    device: str | None = None,
    resolution: int | None = None,
    workers: int = 2,
    threshold: float = DEFAULT_THRESHOLD,
    max_side: int | None = DEFAULT_MAX_SIDE,
    on_stage: Callable[[str], None] | None = None,
) -> dict:
    """Fine-tune, score before and after on the test split, and write the model card."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    sheet = load_datasheet(dataset_dir)
    check_size(size)
    if epochs < 1 or batch_size < 1 or grad_accum < 1:
        raise ValueError("epochs, batch size and accumulation steps must be at least 1")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} already exists; pick a new name")
    try:
        import pytorch_lightning  # noqa: F401
        import rfdetr
    except ImportError as error:
        raise TrainingUnavailable("training needs: uv sync --extra detect --extra train") \
            from error
    device = pick_device(device)
    stage = on_stage or (lambda message: None)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage("Scoring the standard model on the test frames")
    before = _test_score(dataset_dir, size, device, None, threshold, max_side)

    stage(f"Training RF-DETR {size} for {epochs} epochs on {device}")
    started = time.perf_counter()
    model = getattr(rfdetr, SIZES[size])()
    settings = {"epochs": epochs, "batch_size": batch_size, "grad_accum_steps": grad_accum,
                "lr": lr, "num_workers": workers}
    if resolution:
        settings["resolution"] = resolution
    try:
        model.train(dataset_dir=str(dataset_dir), output_dir=str(out_dir / "rfdetr"),
                    device=device, notes={"dataset": sheet["name"], "tool": "soccer-recorder"},
                    **settings)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    train_s = time.perf_counter() - started
    weights = out_dir / WEIGHTS
    if not weights.is_file():
        raise RuntimeError(f"training finished without {WEIGHTS}")
    # Keep the best weights only: resume and per-epoch checkpoints take ~0.5 GB each.
    for extra in (out_dir / "rfdetr").iterdir():
        if extra.suffix in (".pth", ".ckpt") and extra != weights:
            extra.unlink()

    card = {
        "kind": "model",
        "schema_version": 1,
        "name": out_dir.name,
        "id": hashlib.sha256(weights.read_bytes()).hexdigest()[:16],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "family": "RF-DETR",
        "size": size,
        "starting_weights": f"COCO-pretrained RF-DETR {size} (Roboflow), {WEIGHTS_LICENSE}",
        "weights": WEIGHTS,
        "classes": sheet["classes"],
        "dataset": {
            "name": sheet["name"],
            "path": str(dataset_dir.resolve()),
            "datasheet_sha256": hashlib.sha256(
                (dataset_dir / DATASHEET).read_bytes()).hexdigest(),
            "videos": [v["asset_id"] for v in sheet["videos"]],
            "splits": sheet["splits"],
            "split_by": sheet["split_by"],
        },
        "training": {**settings, "device": device, "wall_s": round(train_s, 1),
                     "rfdetr_version": importlib.metadata.version("rfdetr")},
        "use": "Private: trained on footage of children, kept on this computer. Never commit "
               "or share it without the families' permission.",
    }
    (out_dir / MODEL_CARD).write_text(json.dumps(card, indent=2) + "\n")

    stage("Scoring the fine-tuned model on the test frames")
    after = _test_score(dataset_dir, size, device, out_dir, threshold, max_side)
    card["test"] = {"frames": sheet["splits"]["test"]["images"], "threshold": threshold,
                    "before": before, "after": after, "better": _better(before, after)}
    (out_dir / MODEL_CARD).write_text(json.dumps(card, indent=2) + "\n")
    return card
