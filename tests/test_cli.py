import json
from pathlib import Path

import pytest
from conftest import FixedDetector, needs_ffmpeg

from soccer_recorder import cli
from soccer_recorder.cli import main
from soccer_recorder.rfdetr_detector import DetectorUnavailable

pytestmark = needs_ffmpeg


def test_doctor_passes(tmp_path, capsys):
    assert main(["doctor", "--library", str(tmp_path / "lib")]) == 0
    assert "All required checks passed" in capsys.readouterr().out


def test_make_sample_probe_ingest_list(tmp_path, capsys):
    clip = tmp_path / "game.mov"
    library = tmp_path / "lib"
    assert main(["make-sample", str(clip), "--duration", "1", "--width", "320",
                 "--height", "180"]) == 0

    assert main(["probe", str(clip), "--json"]) == 0
    info = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert info["codec"] == "hevc"

    assert main(["ingest", str(clip), "--library", str(library)]) == 0
    assert "Ingested:" in capsys.readouterr().out
    assert main(["ingest", str(clip), "--library", str(library)]) == 0
    assert "Already in library:" in capsys.readouterr().out

    assert main(["list", "--library", str(library)]) == 0
    out = capsys.readouterr().out
    assert "game.mov" in out and "1 video(s)" in out


def test_list_empty_library(tmp_path, capsys):
    assert main(["list", "--library", str(tmp_path)]) == 0
    assert "No videos" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["probe", "ingest"])
def test_bad_input_exits_2_with_message(tmp_path, capsys, command):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("nope")
    assert main([command, str(bogus), *(["--library", str(tmp_path / "lib")]
                                        if command == "ingest" else [])]) == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.fixture
def fixed_detector(monkeypatch):
    made = []

    def make(size="small", *, device=None, threshold=0.3, shape=None):
        detector = FixedDetector()
        detector.name = f"fixed-{size}"
        detector.describe = lambda: {
            "name": detector.name, "device": "cpu",
            "input_shape": list(shape) if shape else [160, 90],
        }
        made.append((size, threshold, shape))
        return detector

    monkeypatch.setattr(cli, "make_detector", make)
    return made


def test_detect_on_a_file(tmp_path, capsys, landscape_clip, fixed_detector):
    out = tmp_path / "run"
    assert main(["detect", str(landscape_clip), "--out", str(out), "--threshold", "0.4"]) == 0
    assert fixed_detector == [("small", 0.4, None)]
    assert (out / "detections.jsonl").is_file() and (out / "annotated.mp4").is_file()
    printed = capsys.readouterr().out
    assert "players       2.0 per frame" in printed


def test_detect_on_an_asset_id_prefix(tmp_path, capsys, landscape_clip, fixed_detector):
    library = tmp_path / "lib"
    assert main(["ingest", str(landscape_clip), "--library", str(library)]) == 0
    asset_id = capsys.readouterr().out.split()[1]
    assert main(["detect", asset_id[:6], "--library", str(library), "--no-video",
                 "--size", "nano"]) == 0
    runs = list((library / asset_id / "runs").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("detect-fixed-nano-")
    assert not (runs[0] / "annotated.mp4").exists()


def test_detect_unknown_target(tmp_path, capsys, fixed_detector):
    assert main(["detect", "nope", "--library", str(tmp_path)]) == 2
    assert "no such file" in capsys.readouterr().err


def test_detect_without_the_extra(tmp_path, capsys, landscape_clip, monkeypatch):
    def unavailable(*args, **kwargs):
        raise DetectorUnavailable("RF-DETR isn't installed. Run: uv sync --extra detect")

    monkeypatch.setattr(cli, "make_detector", unavailable)
    assert main(["detect", str(landscape_clip), "--out", str(tmp_path)]) == 2
    assert "uv sync --extra detect" in capsys.readouterr().err


def test_benchmark_prints_a_row_per_size_and_shape(tmp_path, capsys, landscape_clip,
                                                    fixed_detector):
    out = tmp_path / "bench"
    assert main(["benchmark", str(landscape_clip), "--sizes", "nano,small",
                 "--shapes", "704x384", "--duration", "1", "--out", str(out)]) == 0
    assert [(size, shape) for size, _, shape in fixed_detector] == [
        ("nano", None), ("nano", (704, 384)), ("small", None), ("small", (704, 384)),
    ]
    rows = [line for line in capsys.readouterr().out.splitlines()
            if line.startswith("| fixed-")]
    assert len(rows) == 4
    assert len(json.loads((out / "benchmark.json").read_text())) == 4


def test_sample_frames_then_evaluate(tmp_path, capsys, landscape_clip, fixed_detector):
    frames = tmp_path / "frames"
    assert main(["sample-frames", str(landscape_clip), "--every", "0.5",
                 "--out", str(frames)]) == 0
    names = sorted(p.name for p in frames.glob("*.jpg"))
    assert names[0] == "landscape_00000.jpg" and len(names) == 4
    listing = json.loads((frames / "frames.json").read_text())
    assert [f["t"] for f in listing["frames"]] == [0.0, 0.5, 1.0, 1.5]

    labels = tmp_path / "coco.json"
    labels.write_text(json.dumps({
        "images": [{"id": 1, "file_name": names[0]}],
        "categories": [{"id": 1, "name": "player"}],
        # FixedDetector's first person, in the 320x180 original's pixels.
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 20, 40, 120]}],
    }))
    report_path = tmp_path / "report.json"
    assert main(["evaluate", "--labels", str(labels), "--images", str(frames),
                 "--max-side", "160", "--out", str(report_path)]) == 0
    printed = capsys.readouterr().out
    assert "| person | 1 | 0.5 | 1.0 |" in printed  # second person box is a false positive
    report = json.loads(report_path.read_text())
    assert report["classes"]["person"]["recall"] == 1.0
    assert fixed_detector[-1][1] == 0.05  # evaluate keeps low scores for AP50


def test_evaluate_with_bad_labels(tmp_path, capsys, fixed_detector):
    bad = tmp_path / "coco.json"
    bad.write_text("{}")
    assert main(["evaluate", "--labels", str(bad), "--images", str(tmp_path)]) == 2
    assert "not a COCO labels file" in capsys.readouterr().err


@pytest.mark.skipif(not __import__("importlib").util.find_spec("trackers"),
                    reason="detect extra not installed")
def test_track_then_follow_commands(tmp_path, capsys, monkeypatch):
    from synthetic import OracleDetector, blocker_scene

    scene = blocker_scene()
    scene.duration = 6.0
    video = tmp_path / "scene.mp4"
    truth = scene.write(video)
    library = tmp_path / "lib"
    assert main(["ingest", str(video), "--library", str(library)]) == 0
    asset_id = capsys.readouterr().out.split()[1]
    made = []

    def make(size="small", **kwargs):
        made.append(size)
        return OracleDetector(truth)

    monkeypatch.setattr(cli, "make_detector", make)
    assert main(["track", asset_id, "--library", str(library), "--fps", "15"]) == 0
    out = capsys.readouterr().out
    assert "tracked" in out and "tracks" in out
    runs = library / asset_id / "runs"
    tracks_run = next(runs.glob("tracks-*"))
    assert (tracks_run / "tracks.mp4").is_file() and (tracks_run / "tracks.npz").is_file()

    x1, y1, x2, y2 = scene.players[0].box(1.0)
    point = f"{(x1 + x2) / 2 / 640:.4f},{(y1 + y2) / 2 / 360:.4f}"  # as fractions
    assert main(["follow", asset_id, "--library", str(library), "--fps", "15",
                 "--number", "10", "--name", "Emma", "--at", "1", "--point", point]) == 0
    out = capsys.readouterr().out
    assert "reused from" in out  # the tracking above was reused
    follow_run = next(runs.glob("follow-n10-*"))
    assert (follow_run / "followed.mp4").is_file()
    assert "found" in out and "links       1" in out


def test_follow_command_explains_a_missed_tap(tmp_path, capsys, monkeypatch, landscape_clip):
    pytest.importorskip("trackers")
    monkeypatch.setattr(cli, "make_detector", lambda size="small", **kw: FixedDetector())
    assert main(["follow", str(landscape_clip), "--number", "4", "--at", "1",
                 "--point", "300,170", "--fps", "15"]) == 2
    assert "No player was found where you tapped" in capsys.readouterr().err


@pytest.mark.skipif(not __import__("importlib").util.find_spec("trackers"),
                    reason="detect extra not installed")
def test_follow_then_analyze_with_marked_goals(tmp_path, capsys, monkeypatch):
    from synthetic import OracleDetector, match_audio, match_scene, with_audio

    scene = match_scene()
    truth = scene.write(tmp_path / "silent.mp4")
    match_audio(tmp_path / "sound.wav", scene.duration)
    video = with_audio(tmp_path / "silent.mp4", tmp_path / "sound.wav", tmp_path / "match.mp4")
    library = tmp_path / "lib"
    assert main(["ingest", str(video), "--library", str(library)]) == 0
    asset_id = capsys.readouterr().out.split()[1]
    monkeypatch.setattr(cli, "make_detector", lambda size="small", **kw: OracleDetector(truth))
    x1, y1, x2, y2 = scene.players[0].box(0.5)
    assert main(["follow", asset_id, "--library", str(library), "--number", "10",
                 "--at", "0.5", "--point", f"{(x1 + x2) / 2},{(y1 + y2) / 2}",
                 "--height-cm", "140"]) == 0
    out = capsys.readouterr().out
    assert "passes      1/1 completed, 1 received; 0 shots" in out  # no goals marked yet
    assert "Mark the goals" in out

    width, height = scene.size
    goals = ";".join(",".join(f"{v:.4f}" for v in (g[0] / width, g[1] / height,
                                                     g[2] / width, g[3] / height))
                     for g in scene.goals)
    assert main(["analyze", asset_id, "--library", str(library), "--goals", goals]) == 0
    out = capsys.readouterr().out
    assert "1 shots" in out
    assert "goal        at 6.9 s (95%: ball in the goal, cheering, whistle)" in out
    assert (library / asset_id / "goals.json").is_file()

    assert main(["analyze", asset_id, "--library", str(library), "--goals", "0.1,0.1,0.1"]) == 2
    assert "four fractions of the frame" in capsys.readouterr().err

    # A fix with a second tap, then a highlight reel from the fixed run.
    x1, y1, x2, y2 = scene.players[0].box(5.0)
    assert main(["fix", asset_id, "--library", str(library), "--at", "5",
                 "--point", f"{(x1 + x2) / 2 / width:.4f},{(y1 + y2) / 2 / height:.4f}",
                 "--no-video"]) == 0
    out = capsys.readouterr().out
    assert "taps        2 (the first, plus 1 fix)" in out
    fixed = Path(out.split("outputs     ")[1].strip())
    assert not (fixed / "followed.mp4").exists()
    assert main(["export", asset_id, "--library", str(library), "--vertical"]) == 0
    out = capsys.readouterr().out
    assert "GOAL  0:07" in out and (fixed / "highlights-vertical.mp4").is_file()
    assert main(["export", asset_id, "--library", str(library), "--run", "follow-n1-x"]) == 2


def test_consent_command(tmp_path, capsys, landscape_clip):
    library = tmp_path / "lib"
    assert main(["ingest", str(landscape_clip), "--library", str(library)]) == 0
    asset_id = capsys.readouterr().out.split()[1]
    assert main(["consent", asset_id, "--library", str(library)]) == 0
    assert "no   training our own models" in capsys.readouterr().out
    assert main(["consent", asset_id, "--library", str(library), "--testing", "--training",
                 "--note", "signed forms, folder A"]) == 0
    out = capsys.readouterr().out
    assert "yes  training our own models" in out and "no   showing it publicly" in out
    assert "note  signed forms, folder A" in out
    assert main(["consent", asset_id, "--library", str(library), "--no-training"]) == 0
    assert "no   training our own models" in capsys.readouterr().out


def test_reviews_command_without_any(tmp_path, capsys):
    assert main(["reviews", "--library", str(tmp_path / "lib")]) == 0
    assert "No moments checked yet" in capsys.readouterr().out
