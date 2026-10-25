import pytest
from conftest import DURATION, FPS, needs_ffmpeg

from soccer_recorder import media
from soccer_recorder.media import MediaError, fit_within, probe


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((3840, 2160), 1280, (1280, 720)),
        ((2160, 3840), 1280, (720, 1280)),  # portrait: the long side is the height
        ((1920, 1080), 1280, (1280, 720)),
        ((1280, 720), 1920, (1280, 720)),  # never upscales
        ((1920, 1080), None, (1920, 1080)),
        ((1921, 1081), None, (1920, 1080)),  # rounded down to even
        ((1920, 1080), 1279, (1278, 718)),  # stays within an odd limit
    ],
)
def test_fit_within(size, max_side, expected):
    assert fit_within(*size, max_side) == expected


def test_fit_within_rejects_tiny_limit():
    with pytest.raises(ValueError):
        fit_within(1920, 1080, 1)


def iphone_ffprobe_output(**video_overrides):
    """Trimmed ffprobe JSON from a 4K HDR iPhone recording held in portrait."""
    video = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p10le",
        "color_transfer": "arib-std-b67",
        "r_frame_rate": "30/1",
        # 26.2 fps against a nominal 30: the camera slowed down in a dim gym.
        "avg_frame_rate": "6000/229",
        "nb_frames": "1446",
        "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
        "tags": {"creation_time": "2026-10-18T15:02:11.000000Z"},
    }
    video.update(video_overrides)
    return {
        "streams": [
            video,
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "55.200000",
            "bit_rate": "48000000",
            "tags": {
                "com.apple.quicktime.make": "Apple",
                "com.apple.quicktime.model": "iPhone 15",
                "com.apple.quicktime.creationdate": "2026-10-18T08:02:11-0700",
            },
        },
    }


@pytest.fixture
def fake_file(tmp_path):
    path = tmp_path / "IMG_4821.MOV"
    path.write_bytes(b"not really a video")
    return path


def test_probe_reads_iphone_metadata(monkeypatch, fake_file):
    monkeypatch.setattr(media, "run_ffprobe", lambda path: iphone_ffprobe_output())
    info = probe(fake_file)

    assert info.codec == "hevc"
    assert (info.width, info.height) == (3840, 2160)
    assert info.rotation == 90
    assert (info.display_width, info.display_height) == (2160, 3840)
    assert info.duration_s == pytest.approx(55.2)
    assert info.avg_frame_rate == "6000/229"
    assert info.is_vfr
    assert info.is_hdr
    assert info.has_audio
    assert info.device == "Apple iPhone 15"
    assert info.creation_time == "2026-10-18T08:02:11-0700"
    assert info.frame_count == 1446
    assert len(info.warnings) == 3  # HDR, VFR, portrait


def test_probe_reads_legacy_rotate_tag(monkeypatch, fake_file):
    output = iphone_ffprobe_output(side_data_list=[], tags={"rotate": "270"})
    monkeypatch.setattr(media, "run_ffprobe", lambda path: output)
    info = probe(fake_file)
    assert info.rotation == 270
    assert (info.display_width, info.display_height) == (2160, 3840)


def test_probe_sdr_constant_rate_landscape_has_no_warnings(monkeypatch, fake_file):
    # 29.946 against 30 is within tolerance: a rounding difference, not VFR.
    output = iphone_ffprobe_output(
        side_data_list=[], color_transfer="bt709", avg_frame_rate="16500/551"
    )
    monkeypatch.setattr(media, "run_ffprobe", lambda path: output)
    info = probe(fake_file)
    assert (info.rotation, info.is_hdr, info.is_vfr) == (0, False, False)
    assert info.warnings == []


def test_probe_skips_cover_art_stream(monkeypatch, fake_file):
    output = iphone_ffprobe_output()
    cover = {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
             "disposition": {"attached_pic": 1}}
    output["streams"].insert(0, cover)
    monkeypatch.setattr(media, "run_ffprobe", lambda path: output)
    assert probe(fake_file).codec == "hevc"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda out: out["streams"].pop(0), "no video stream"),
        (lambda out: out["format"].update(duration="0"), "no duration"),
        (lambda out: out["format"].update(duration="N/A"), "no duration"),
        (lambda out: out["streams"][0].update(avg_frame_rate="0/0", r_frame_rate="0/0"),
         "no frame rate"),
    ],
)
def test_probe_rejects_unusable_metadata(monkeypatch, fake_file, change, message):
    output = iphone_ffprobe_output()
    change(output)
    monkeypatch.setattr(media, "run_ffprobe", lambda path: output)
    with pytest.raises(MediaError, match=message):
        probe(fake_file)


def test_probe_missing_file(tmp_path):
    with pytest.raises(MediaError, match="No such file"):
        probe(tmp_path / "missing.mov")


@needs_ffmpeg
def test_probe_rejects_non_video(fake_file):
    with pytest.raises(MediaError, match="could not read"):
        probe(fake_file)


@needs_ffmpeg
def test_probe_landscape_clip(landscape_clip):
    info = probe(landscape_clip)
    assert info.codec == "h264"
    assert (info.display_width, info.display_height) == (320, 180)
    assert info.rotation == 0
    assert info.fps == pytest.approx(FPS)
    assert info.duration_s == pytest.approx(DURATION, abs=0.1)
    assert info.has_audio
    assert not info.is_hdr and not info.is_vfr
    assert info.warnings == []


@needs_ffmpeg
def test_probe_portrait_clip_swaps_display_size(portrait_clip):
    info = probe(portrait_clip)
    assert info.codec == "hevc"
    assert (info.width, info.height) == (320, 180)
    assert info.rotation in (90, 270)
    assert (info.display_width, info.display_height) == (180, 320)
    assert any("portrait" in warning for warning in info.warnings)
