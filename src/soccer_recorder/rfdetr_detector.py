"""RF-DETR person and ball detector.

Needs the `detect` extra (`uv sync --extra detect`). The first use of each
size downloads its COCO-pretrained weights (about 370 MB for small) from
Roboflow's public storage into ~/.roboflow/models/.

Only the sizes whose weights are Apache-2.0 can be loaded (decisions.md T10).
RF-DETR's Atto, Femto, Pico, XL and 2XL weights are under Roboflow's PML 1.0
and are refused here.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Mapping, Sequence

import numpy as np

from .detect import Box

# Let the few PyTorch operations Apple's GPU lacks run on the CPU instead of
# failing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SIZES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
WEIGHTS_LICENSE = "Apache-2.0"
DEFAULT_SIZE = "small"
DEFAULT_THRESHOLD = 0.3


class DetectorUnavailable(Exception):
    """RF-DETR (or PyTorch) is not installed."""


def check_size(size: str) -> str:
    if size not in SIZES:
        raise ValueError(
            f"RF-DETR size {size!r} isn't allowed: only {', '.join(SIZES)} have "
            f"{WEIGHTS_LICENSE} weights (see docs/decisions.md T10)"
        )
    return size


def parse_shape(text: str | None) -> tuple[int, int] | None:
    """'704x384' (width x height, like video sizes) -> (704, 384)."""
    if not text:
        return None
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"shape must look like 704x384 (width x height), not {text!r}") from None
    if width <= 0 or height <= 0:
        raise ValueError("shape sides must be positive")
    return width, height


def pick_device(requested: str | None = None) -> str:
    """Use what was asked for, else Apple's GPU, else CUDA, else the CPU."""
    if requested and requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def label_names(
    detections, class_ids: Sequence[int], fallback: Mapping[int, str] | None = None
) -> list[str]:
    """Class names for RF-DETR detections.

    Use the names RF-DETR resolved itself (`detections.data["class_name"]`).
    Don't index `model.class_names` with the ids: COCO-pretrained models emit
    the original COCO ids (person = 1, sports ball = 37), but `class_names` is
    a 0-based list, so that lookup turns every person into a "bicycle".
    """
    names = getattr(detections, "data", {}).get("class_name")
    if names is not None and len(names) == len(class_ids):
        return [str(name) for name in names]
    if fallback is None:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        fallback = COCO_CLASSES
    return [fallback.get(int(class_id), str(int(class_id))) for class_id in class_ids]


class RFDETRDetector:
    """Wraps an RF-DETR model as a `Detector` (BGR in, boxes in image pixels out)."""

    def __init__(
        self,
        size: str = DEFAULT_SIZE,
        *,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        shape: tuple[int, int] | None = None,
    ):
        check_size(size)
        try:
            import rfdetr
        except ImportError as error:
            raise DetectorUnavailable(
                "RF-DETR isn't installed. Run: uv sync --extra detect"
            ) from error
        self.size = size
        self.name = f"rfdetr-{size}"
        self.threshold = threshold
        self.device = pick_device(device)
        self.model = getattr(rfdetr, SIZES[size])(device=self.device)

        config = self.model.model_config
        self.block = config.patch_size * getattr(config, "num_windows", 1)
        self.resolution = config.resolution
        if shape is not None:
            bad = [side for side in shape if side % self.block]
            if bad:
                raise ValueError(
                    f"shape sides must be multiples of {self.block} for RF-DETR {size}; "
                    f"{shape[0]}x{shape[1]} isn't"
                )
        self.shape = shape

    def detect(self, image: np.ndarray) -> list[Box]:
        rgb = np.ascontiguousarray(image[..., ::-1])
        detections = self.model.predict(
            rgb,
            threshold=self.threshold,
            shape=(self.shape[1], self.shape[0]) if self.shape else None,
            include_source_image=False,
        )
        names = label_names(detections, detections.class_id)
        return [
            Box(float(x1), float(y1), float(x2), float(y2), float(score), name)
            for (x1, y1, x2, y2), score, name in zip(
                detections.xyxy, detections.confidence, names, strict=True
            )
        ]

    def describe(self) -> dict:
        import torch

        width, height = self.shape or (self.resolution, self.resolution)
        return {
            "name": self.name,
            "family": "RF-DETR",
            "size": self.size,
            "weights": "COCO-pretrained (Roboflow)",
            "weights_license": WEIGHTS_LICENSE,
            "input_shape": [width, height],
            "threshold": self.threshold,
            "device": self.device,
            "rfdetr_version": importlib.metadata.version("rfdetr"),
            "torch_version": torch.__version__,
        }
