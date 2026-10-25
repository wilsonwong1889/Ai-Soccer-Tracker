import json

import pytest
from conftest import needs_ffmpeg

from soccer_recorder import library as library_module
from soccer_recorder.library import (
    IngestError,
    default_library,
    file_sha256,
    ingest,
    list_assets,
    load_asset,
)
from soccer_recorder.media import probe

pytestmark = needs_ffmpeg


def test_ingest_copies_source_and_writes_manifest(tmp_path, landscape_clip):
    asset, created = ingest(landscape_clip, tmp_path)

    assert created
    assert asset.asset_id == file_sha256(landscape_clip)[:16]
    assert asset.directory == tmp_path / asset.asset_id
    assert asset.source_path.read_bytes() == landscape_clip.read_bytes()
    assert not asset.source_path.is_symlink()
    assert asset.proxy is None

    manifest = json.loads((asset.directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["original_filename"] == "landscape.mp4"
    assert manifest["source"] == {"path": "source.mp4", "mode": "copy"}
    assert manifest["video"]["display_width"] == 320
    # No absolute paths in the manifest: they can carry names (e.g. a child's).
    assert str(landscape_clip.parent) not in json.dumps(manifest)


def test_ingesting_the_same_file_again_returns_the_existing_asset(tmp_path, landscape_clip):
    first, _ = ingest(landscape_clip, tmp_path)
    second, created = ingest(landscape_clip, tmp_path)

    assert not created
    assert second.asset_id == first.asset_id
    assert second.ingested_at == first.ingested_at
    assert [p.name for p in tmp_path.iterdir()] == [first.asset_id]


def test_ingest_link_mode_symlinks_the_original(tmp_path, portrait_clip):
    asset, _ = ingest(portrait_clip, tmp_path, link=True)
    assert asset.source_path.is_symlink()
    assert asset.source_path.resolve() == portrait_clip.resolve()
    assert asset.source == "source.mov"  # extension lower-cased
    assert asset.source_mode == "symlink"


def test_ingest_with_proxy(tmp_path, landscape_clip):
    asset, _ = ingest(landscape_clip, tmp_path, proxy=True, proxy_max_side=160)

    assert asset.proxy["width"] == 160 and asset.proxy["height"] == 90
    proxy = probe(asset.proxy_path)
    assert proxy.codec == "h264"
    assert (proxy.display_width, proxy.display_height) == (160, 90)
    assert not proxy.has_audio
    assert proxy.duration_s == pytest.approx(asset.video.duration_s, abs=0.15)
    assert load_asset(asset.directory).proxy == asset.proxy
    assert not list(asset.directory.glob(".*.part"))


def test_proxy_of_rotated_video_is_upright(tmp_path, portrait_clip):
    asset, _ = ingest(portrait_clip, tmp_path, proxy=True)
    proxy = probe(asset.proxy_path)
    assert (proxy.display_width, proxy.display_height) == (180, 320)
    assert proxy.rotation == 0


def test_reingest_adds_a_missing_proxy(tmp_path, landscape_clip):
    ingest(landscape_clip, tmp_path)
    asset, created = ingest(landscape_clip, tmp_path, proxy=True, proxy_max_side=160)
    assert not created
    assert asset.proxy_path.is_file()


def test_ingest_rejects_non_video_without_writing_anything(tmp_path):
    bogus = tmp_path / "notes.mov"
    bogus.write_text("this is not a video")
    library = tmp_path / "library"
    with pytest.raises(IngestError):
        ingest(bogus, library)
    assert not library.exists()


def test_failed_ingest_leaves_no_partial_asset(tmp_path, landscape_clip, monkeypatch):
    def broken_proxy(asset, max_side):
        raise IngestError("encoder crashed")

    monkeypatch.setattr(library_module, "make_proxy", broken_proxy)
    with pytest.raises(IngestError, match="encoder crashed"):
        ingest(landscape_clip, tmp_path, proxy=True)
    assert list(tmp_path.iterdir()) == []


def test_list_assets(tmp_path, landscape_clip, portrait_clip):
    assert list_assets(tmp_path / "missing") == []
    ingest(landscape_clip, tmp_path)
    ingest(portrait_clip, tmp_path, link=True)
    (tmp_path / ".ingest-leftover").mkdir()  # staging folders are ignored

    names = [asset.original_filename for asset in list_assets(tmp_path)]
    assert sorted(names) == ["IMG_0001.MOV", "landscape.mp4"]


def test_unknown_manifest_schema_is_refused(tmp_path, landscape_clip):
    asset, _ = ingest(landscape_clip, tmp_path)
    manifest = asset.directory / "manifest.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = 99
    manifest.write_text(json.dumps(data))
    with pytest.raises(IngestError, match="schema 99"):
        load_asset(asset.directory)


def test_default_library_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_RECORDER_LIBRARY", str(tmp_path))
    assert default_library() == tmp_path
    monkeypatch.delenv("SOCCER_RECORDER_LIBRARY")
    assert str(default_library()) == "data/library"
