import importlib.util
import json
import os

import pytest
from conftest import needs_ffmpeg
from labelled import labelled_game
from synthetic import blocker_scene, crossing_scene, match_scene

from soccer_recorder.dataset import DatasetError, build_dataset, load_datasheet
from soccer_recorder.library import set_consent

pytestmark = needs_ffmpeg


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    """Three labelled synthetic games in one library, none with permission yet."""
    root = tmp_path_factory.mktemp("games")
    library = root / "lib"
    made = []
    for k, make in enumerate((blocker_scene, crossing_scene, match_scene)):
        scene = make()
        scene.duration = 4.0
        made.append(labelled_game(root / f"g{k}", library, scene))
    return root, library, made


def allow(asset, training=True):
    set_consent(asset, training=training, note="synthetic game")


def test_nothing_is_used_without_training_permission(games, tmp_path):
    _, library, made = games
    for asset, _, _ in made:
        allow(asset, False)
    set_consent(made[0][0], testing=True, public=True)  # other permissions don't count
    with pytest.raises(DatasetError, match="training permission"):
        build_dataset([(labels, images) for _, labels, images in made], tmp_path / "ds",
                      library=library)
    assert not (tmp_path / "ds").exists()


def test_one_game_is_split_in_time_and_others_are_left_out(games, tmp_path):
    _, library, made = games
    for k, (asset, _, _) in enumerate(made):
        allow(asset, k == 0)
    sheet = build_dataset([(labels, images) for _, labels, images in made], tmp_path / "ds",
                          library=library)
    assert sheet["split_by"] == "time"
    assert [v["asset_id"] for v in sheet["videos"]] == [made[0][0].asset_id]
    frames = sheet["videos"][0]["frames"]
    assert sum(sheet["splits"][s]["images"] for s in ("train", "valid", "test")) == frames
    assert sheet["splits"]["train"]["images"] > sheet["splits"]["test"]["images"] > 0
    left_out = {r["asset_id"]: r["frames"] for r in sheet["refused"]["no training permission"]}
    assert set(left_out) == {made[1][0].asset_id, made[2][0].asset_id}
    assert sheet["videos"][0]["consent"]["training"] is True

    coco = json.loads((tmp_path / "ds" / "train" / "_annotations.coco.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["person", "sports ball"]
    # Every frame has its players plus the spectator, all as people.
    first = coco["images"][0]
    people = [a for a in coco["annotations"] if a["image_id"] == first["id"]]
    assert len(people) >= 3 and {a["category_id"] for a in people} <= {1, 2}
    image = tmp_path / "ds" / "train" / first["file_name"]
    assert image.is_file() and os.stat(image).st_nlink >= 2  # linked, not copied
    assert "synthetic game" in (tmp_path / "ds" / "datasheet.md").read_text()
    assert load_datasheet(tmp_path / "ds")["name"] == "ds"


def test_three_games_are_split_by_game(games, tmp_path):
    _, library, made = games
    for asset, _, _ in made:
        allow(asset)
    sheet = build_dataset([(labels, images) for _, labels, images in made], tmp_path / "ds",
                          library=library)
    assert sheet["split_by"] == "game"
    for video in sheet["videos"]:
        assert sorted(video["splits"].values())[:2] == [0, 0]  # each game in one split only
    assert all(sheet["splits"][s]["images"] for s in ("train", "valid", "test"))


def test_datasets_are_never_overwritten_and_labels_are_checked(games, tmp_path):
    _, library, made = games
    allow(made[0][0])
    sources = [(made[0][1], made[0][2])]
    build_dataset(sources, tmp_path / "ds", library=library)
    with pytest.raises(DatasetError, match="already exists"):
        build_dataset(sources, tmp_path / "ds", library=library)
    with pytest.raises(DatasetError, match="labelled twice"):
        build_dataset(sources * 2, tmp_path / "ds2", library=library)
    odd = json.loads(made[0][1].read_text())
    odd["categories"].append({"id": 9, "name": "dog"})
    (tmp_path / "odd.json").write_text(json.dumps(odd))
    with pytest.raises(DatasetError, match="dog"):
        build_dataset([(tmp_path / "odd.json", made[0][2])], tmp_path / "ds3", library=library)


def test_frames_not_from_the_library_are_left_out(games, tmp_path):
    _, library, made = games
    for asset, _, _ in made:
        allow(asset)
    stray = json.loads(made[0][1].read_text())
    stray["images"] = [{**image, "id": image["id"], "file_name": f"phone_{image['id']}.jpg"}
                       for image in stray["images"]]
    (tmp_path / "stray.json").write_text(json.dumps(stray))
    sources = [(labels, images) for _, labels, images in made] + [
        (tmp_path / "stray.json", made[0][2])]
    sheet = build_dataset(sources, tmp_path / "ds", library=library)
    assert sheet["refused"]["not from a library video"] == len(stray["images"])


def test_training_needs_a_datasheet(tmp_path):
    from soccer_recorder.finetune import load_model_card, train_detector

    (tmp_path / "loose" / "train").mkdir(parents=True)
    with pytest.raises(DatasetError, match="datasheet"):
        train_detector(tmp_path / "loose", tmp_path / "model")
    with pytest.raises(ValueError, match="model.json"):
        load_model_card(tmp_path / "loose")


@pytest.mark.skipif(not os.environ.get("SOCCER_RECORDER_TRAIN_TESTS"),
                    reason="set SOCCER_RECORDER_TRAIN_TESTS=1 to fine-tune for real (minutes)")
@pytest.mark.skipif(not importlib.util.find_spec("pytorch_lightning"),
                    reason="train extra not installed")
def test_fine_tune_then_detect_with_the_new_weights(games, tmp_path):
    from soccer_recorder.finetune import train_detector
    from soccer_recorder.frames import read_image
    from soccer_recorder.rfdetr_detector import RFDETRDetector

    _, library, made = games
    for asset, _, _ in made:
        allow(asset)
    build_dataset([(labels, images) for _, labels, images in made], tmp_path / "ds",
                  library=library)
    card = train_detector(tmp_path / "ds", tmp_path / "model", size="nano", epochs=1,
                          batch_size=2, grad_accum=1, workers=0, device="cpu")
    assert card["test"]["before"]["person"]["ap50"] is not None
    assert set(card["dataset"]["videos"]) == {asset.asset_id for asset, _, _ in made}
    kept = sorted(p.name for p in (tmp_path / "model" / "rfdetr").iterdir()
                  if p.suffix in (".pth", ".ckpt"))
    assert kept == ["checkpoint_best_total.pth"]  # resume checkpoints (~0.5 GB each) removed
    detector = RFDETRDetector(weights=tmp_path / "model", device="cpu", threshold=0.01)
    assert detector.name.startswith("rfdetr-nano-ft")
    image, _, _ = read_image(next((tmp_path / "ds" / "test").glob("*.jpg")), None)
    assert {box.label for box in detector.detect(image)} <= {"person", "sports ball"}
