"""Synthetic games with labelled frames, as CVAT would export them (for training tests)."""

from __future__ import annotations

import json
from pathlib import Path

from synthetic import BALL_KEY

from soccer_recorder.frames import sample_frames
from soccer_recorder.library import ingest

CATEGORIES = [{"id": 1, "name": "player"}, {"id": 2, "name": "ball"},
              {"id": 3, "name": "spectator"}]


def labelled_game(root: Path, library: Path, scene, every_s: float = 0.5):
    """Write the scene, ingest it, sample frames and label them from the truth.

    Returns (asset, labels file, images folder).
    """
    root.mkdir(parents=True, exist_ok=True)
    truth = scene.write(root / "game.mp4")
    asset, _ = ingest(root / "game.mp4", library)
    images = root / "frames"
    frames = sample_frames(asset.source_path, images, every_s=every_s, prefix=asset.asset_id)
    width, height = scene.size
    coco = {"categories": CATEGORIES, "images": [], "annotations": []}
    for n, frame in enumerate(frames, start=1):
        i = min(round(frame["t"] * scene.fps), len(scene.times) - 1)
        coco["images"].append({"id": n, "file_name": frame["file"], "width": width,
                               "height": height})
        for name, boxes in truth.items():
            box = boxes[i]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": n,
                "category_id": 2 if name == BALL_KEY else 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0,
            })
        coco["annotations"].append({  # a parent on the touchline
            "id": len(coco["annotations"]) + 1, "image_id": n, "category_id": 3,
            "bbox": [5, 5, 12, 30], "iscrowd": 0,
        })
    labels = root / "instances_default.json"
    labels.write_text(json.dumps(coco))
    return asset, labels, images
