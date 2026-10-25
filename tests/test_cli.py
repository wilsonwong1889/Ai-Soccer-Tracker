import json

import pytest
from conftest import needs_ffmpeg

from soccer_recorder.cli import main

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
