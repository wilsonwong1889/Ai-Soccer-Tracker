"""Training sets from labelled frames, only from videos the families agreed to.

The input is what the R2 labelling loop already produces (docs/labeling.md):
JPEGs from `sample-frames`, named `<asset id>_<n>.jpg`, and a COCO file
exported from CVAT. The asset id in each name says which library video a
frame came from, and that video's manifest says whether training is allowed.
Frames from any other video, or from one without training permission, are
left out and counted in the datasheet. There is no way to override this.

The output is the layout RF-DETR trains from:

    data/datasets/<name>/
      train/  valid/  test/      JPEGs + _annotations.coco.json in each
      datasheet.json             where every frame came from, and the consent
      datasheet.md               the same, to read (and for the AI disclosure)

Labels become the detector's own two classes, so a fine-tuned model is a
drop-in replacement: player, goalkeeper, referee and spectator are all
"person"; ball is "sports ball".

Frames are split by game when there are at least three games, so the test
score comes from games the model never saw. With fewer, each game is split
in time (first 70% train, next 15% valid, last 15% test).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .library import Asset, list_assets

CLASSES = ("person", "sports ball")
TO_CLASS = {
    "player": "person", "goalkeeper": "person", "referee": "person", "spectator": "person",
    "person": "person", "ball": "sports ball", "sports ball": "sports ball",
}
SPLITS = ("train", "valid", "test")
DATASHEET = "datasheet.json"
FRAME_NAME = re.compile(r"([0-9a-f]{16})_(\d+)\.jpe?g", re.IGNORECASE)
MIN_GAMES_TO_SPLIT_BY_GAME = 3


class DatasetError(ValueError):
    """The training set can't be built from these labels."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_labels(labels: Path) -> list[dict]:
    """Each labelled image: file name, size and boxes (class, x, y, w, h, crowd)."""
    try:
        data = json.loads(Path(labels).read_text())
        names = {c["id"]: str(c["name"]).strip().lower() for c in data["categories"]}
        images = {i["id"]: i for i in data["images"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DatasetError(f"{labels}: not a COCO labels file ({error})") from error
    unknown = sorted(set(names.values()) - set(TO_CLASS) - {"ignore", "crowd"})
    if unknown:
        raise DatasetError(f"{labels}: labels {', '.join(unknown)} aren't known; use "
                           f"{', '.join(sorted(TO_CLASS))} (docs/labeling.md)")
    found = {i: {"file": Path(image["file_name"]).name, "width": image.get("width"),
                 "height": image.get("height"), "boxes": []} for i, image in images.items()}
    for annotation in data.get("annotations", []):
        try:
            name = names[annotation["category_id"]]
            x, y, w, h = (float(v) for v in annotation["bbox"])
            item = found[annotation["image_id"]]
        except (KeyError, ValueError, TypeError) as error:
            raise DatasetError(f"{labels}: bad annotation {annotation.get('id')}: "
                               f"{error}") from error
        if w <= 0 or h <= 0:
            continue
        crowd = name in ("ignore", "crowd") or bool(annotation.get("iscrowd"))
        item["boxes"].append({"class": TO_CLASS.get(name, "person"), "bbox": [x, y, w, h],
                              "crowd": crowd})
    return list(found.values())


def _split(frames: list[dict], seed: int) -> tuple[str, dict[str, str]]:
    """Which split each frame goes in: by game when there are enough games."""
    games = sorted({f["asset_id"] for f in frames})
    where: dict[str, str] = {}
    if len(games) >= MIN_GAMES_TO_SPLIT_BY_GAME:
        order = games[:]
        random.Random(seed).shuffle(order)
        n_test = max(1, round(0.15 * len(order)))
        n_valid = max(1, round(0.15 * len(order)))
        split_of = {g: "test" for g in order[:n_test]}
        split_of.update({g: "valid" for g in order[n_test:n_test + n_valid]})
        for frame in frames:
            where[frame["file"]] = split_of.get(frame["asset_id"], "train")
        return "game", where
    for game in games:
        mine = sorted((f for f in frames if f["asset_id"] == game), key=lambda f: f["index"])
        n = len(mine)
        for k, frame in enumerate(mine):
            share = k / n
            where[frame["file"]] = "train" if share < 0.7 else "valid" if share < 0.85 else "test"
    return "time", where


def build_dataset(sources: list[tuple[Path, Path]], out_dir: Path, *,
                  library: Path | None = None, seed: int = 0) -> dict:
    """Build a training set from (labels file, images folder) pairs. Returns the datasheet."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DatasetError(f"{out_dir} already exists; pick a new name (datasets are never "
                           "overwritten, so models stay traceable to what they learnt from)")
    assets: dict[str, Asset] = {a.asset_id: a for a in list_assets(library)}
    frames, refused = [], {"no training permission": {}, "not from a library video": 0,
                           "image file missing": 0}
    for labels, images in sources:
        for item in _read_labels(labels):
            match = FRAME_NAME.fullmatch(item["file"])
            asset = assets.get(match[1]) if match else None
            if asset is None:
                refused["not from a library video"] += 1
                continue
            if not asset.allows("training"):
                bucket = refused["no training permission"]
                bucket[asset.asset_id] = bucket.get(asset.asset_id, 0) + 1
                continue
            path = Path(images) / item["file"]
            if not path.is_file():
                refused["image file missing"] += 1
                continue
            frames.append({**item, "path": path, "asset_id": asset.asset_id,
                           "index": int(match[2])})
    if not frames:
        raise DatasetError("no labelled frames from videos with training permission. Record "
                           "permission with `soccer-recorder consent <asset> --training` (only "
                           "if every family agreed)")
    if len({f["file"] for f in frames}) != len(frames):
        raise DatasetError("the same frame is labelled twice; give each labels file once")
    split_by, where = _split(frames, seed)
    counts = {s: sum(where[f["file"]] == s for f in frames) for s in SPLITS}
    if not all(counts.values()):
        raise DatasetError(f"not enough labelled frames to fill train, valid and test "
                           f"({counts}); label more frames")

    staging = out_dir.with_name(f".{out_dir.name}-building")
    shutil.rmtree(staging, ignore_errors=True)
    categories = [{"id": k + 1, "name": name, "supercategory": "none"}
                  for k, name in enumerate(CLASSES)]
    class_id = {c["name"]: c["id"] for c in categories}
    stats = {s: {"images": 0, **{c: 0 for c in CLASSES}} for s in SPLITS}
    for split in SPLITS:
        folder = staging / split
        folder.mkdir(parents=True)
        coco = {"info": {"description": f"soccer-recorder {split}"}, "categories": categories,
                "images": [], "annotations": []}
        for frame in (f for f in frames if where[f["file"]] == split):
            target = folder / frame["file"]
            try:
                os.link(frame["path"], target)  # same disk: no second copy of the frames
            except OSError:
                shutil.copy2(frame["path"], target)
            width, height = frame["width"], frame["height"]
            if not (width and height):
                from .frames import read_image

                height, width = read_image(target, max_side=None).shape[:2]
            image_id = len(coco["images"]) + 1
            coco["images"].append({"id": image_id, "file_name": frame["file"],
                                   "width": width, "height": height})
            for box in frame["boxes"]:
                x, y, w, h = box["bbox"]
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": image_id,
                    "category_id": class_id[box["class"]], "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": int(box["crowd"]),
                })
                if not box["crowd"]:
                    stats[split][box["class"]] += 1
            stats[split]["images"] += 1
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))

    videos = []
    for asset_id in sorted({f["asset_id"] for f in frames}):
        consent = assets[asset_id].consent or {}
        mine = [f for f in frames if f["asset_id"] == asset_id]
        videos.append({
            "asset_id": asset_id,
            "frames": len(mine),
            "splits": {s: sum(where[f["file"]] == s for f in mine) for s in SPLITS},
            "consent": {k: consent.get(k) for k in ("testing", "training", "public", "note",
                                                    "updated_at")},
        })
    sheet = {
        "kind": "datasheet",
        "schema_version": 1,
        "name": out_dir.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "classes": list(CLASSES),
        "split_by": split_by,
        "splits": stats,
        "videos": videos,
        "refused": {**refused, "no training permission": [
            {"asset_id": k, "frames": v} for k, v in refused["no training permission"].items()]},
        "labels_files": [{"file": Path(labels).name, "sha256": _sha256(labels)}
                         for labels, _ in sources],
        "rules": "Only frames from library videos whose families agreed to training. "
                 "No footage from public datasets. Labels by hand in CVAT.",
    }
    (staging / DATASHEET).write_text(json.dumps(sheet, indent=2) + "\n")
    (staging / "datasheet.md").write_text(datasheet_markdown(sheet))
    if out_dir.exists():
        out_dir.rmdir()  # empty, checked above
    staging.rename(out_dir)
    return sheet


def datasheet_markdown(sheet: dict) -> str:
    """The datasheet as a page to read (and to quote in the AI disclosure)."""
    lines = [
        f"# Datasheet: {sheet['name']}",
        "",
        f"Built {sheet['created_at']} by soccer-recorder {sheet['tool_version']}.",
        "",
        "## What it contains",
        "",
        "| Split | Images | People | Balls |",
        "| --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        s = sheet["splits"][split]
        lines.append(f"| {split} | {s['images']} | {s['person']} | {s['sports ball']} |")
    lines += [
        "",
        f"Split by **{sheet['split_by']}**"
        + (" (test frames come from games not used for training)." if sheet["split_by"] == "game"
           else " (fewer than three games: each game's last 15% is the test set)."),
        "",
        "## Where it came from",
        "",
        "Frames from our own recordings only, each video recorded with the "
        "families' permission to use it for training:",
        "",
        "| Video (asset id) | Frames | Permission recorded | Note |",
        "| --- | --- | --- | --- |",
    ]
    for video in sheet["videos"]:
        consent = video["consent"]
        lines.append(f"| {video['asset_id']} | {video['frames']} | "
                     f"{consent.get('updated_at') or '?'} | {consent.get('note') or ''} |")
    refused = sheet["refused"]
    no_permission = sum(r["frames"] for r in refused["no training permission"])
    lines += [
        "",
        "Left out: "
        f"{no_permission} frames from videos without training permission, "
        f"{refused['not from a library video']} not from a library video, "
        f"{refused['image file missing']} with the image missing.",
        "",
        "Labels come from the files listed in datasheet.json, drawn by hand in CVAT "
        "(docs/labeling.md). Player, goalkeeper, referee and spectator boxes are all "
        "\"person\"; ball is \"sports ball\".",
        "",
    ]
    return "\n".join(lines)


def load_datasheet(dataset_dir: Path) -> dict:
    path = Path(dataset_dir) / DATASHEET
    if not path.is_file():
        raise DatasetError(f"{dataset_dir} has no {DATASHEET}: only training sets made by "
                           "`soccer-recorder dataset` can be trained on, so every frame's "
                           "permission is known")
    return json.loads(path.read_text())
