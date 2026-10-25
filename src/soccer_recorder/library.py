"""A local library of ingested videos.

Each video gets a folder named after its content hash, so ingesting the same
file twice finds the first copy instead of adding a second. Later releases
key processing jobs and minute metering on that id: a retried upload or job
points at the same asset, never a new one.

    data/library/
      3f9a1c0e5b7d2a64/
        manifest.json   what the file is (VideoInfo), where it came from, and
                        what the families agreed it may be used for
        source.mov      copy of the original, or a symlink to it with --link
        proxy.mp4       optional small constant-rate copy for detection

Permission ("consent") is recorded per video, for each use separately,
because every family whose child can be identified has to agree:

- testing:  measuring how accurate the software is (labelled frames, reviews);
- training: teaching our own models (fine-tuning the detector, re-ID);
- public:   showing it outside the family, e.g. in the campaign video.

A new video has none of them: it's only for watching on this computer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .frames import DEFAULT_MAX_SIDE
from .media import MediaError, VideoInfo, fit_within, probe, require_tool

SCHEMA_VERSION = 1
LIBRARY_ENV = "SOCCER_RECORDER_LIBRARY"
DEFAULT_LIBRARY = Path("data/library")
ASSET_ID_LENGTH = 16
MANIFEST = "manifest.json"
PROXY = "proxy.mp4"
HASH_CHUNK = 8 * 1024 * 1024
USES = ("testing", "training", "public")


class IngestError(Exception):
    """A video could not be added to the library."""


@dataclass
class Asset:
    directory: Path
    asset_id: str
    sha256: str
    original_filename: str
    ingested_at: str
    source: str
    source_mode: str
    video: VideoInfo
    proxy: dict | None = None
    consent: dict | None = None  # {"testing": bool, "training": bool, "public": bool, ...}

    def allows(self, use: str) -> bool:
        """Whether the families agreed to this use of the video."""
        if use not in USES:
            raise ValueError(f"use must be one of {', '.join(USES)}")
        return bool((self.consent or {}).get(use))

    @property
    def source_path(self) -> Path:
        return self.directory / self.source

    @property
    def proxy_path(self) -> Path | None:
        return self.directory / self.proxy["path"] if self.proxy else None

    def to_manifest(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "original_filename": self.original_filename,
            "ingested_at": self.ingested_at,
            "source": {"path": self.source, "mode": self.source_mode},
            "video": self.video.to_dict(),
            "proxy": self.proxy,
            "consent": self.consent,
        }


def default_library() -> Path:
    return Path(os.environ.get(LIBRARY_ENV) or DEFAULT_LIBRARY)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_asset(directory: Path) -> Asset:
    data = json.loads((directory / MANIFEST).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise IngestError(
            f"{directory.name}: manifest schema {data.get('schema_version')} is not "
            f"supported (expected {SCHEMA_VERSION})"
        )
    return Asset(
        directory=directory,
        asset_id=data["asset_id"],
        sha256=data["sha256"],
        original_filename=data["original_filename"],
        ingested_at=data["ingested_at"],
        source=data["source"]["path"],
        source_mode=data["source"]["mode"],
        video=VideoInfo.from_dict(data["video"]),
        proxy=data.get("proxy"),
        consent=data.get("consent"),
    )


def set_consent(asset: Asset, *, note: str | None = None, **uses: bool) -> dict:
    """Record what the families agreed this video may be used for.

    Only the uses passed change. The note says where the signed permission is
    kept (e.g. "forms from all 12 families, folder Team 2026"); never write
    children's names in it.
    """
    unknown = set(uses) - set(USES)
    if unknown:
        raise ValueError(f"unknown use {', '.join(sorted(unknown))}; uses are {', '.join(USES)}")
    consent = {use: bool((asset.consent or {}).get(use)) for use in USES}
    consent.update({use: bool(value) for use, value in uses.items()})
    if note is not None:
        consent["note"] = note.strip()[:300]
    elif asset.consent and asset.consent.get("note"):
        consent["note"] = asset.consent["note"]
    consent["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    asset.consent = consent
    _write_manifest(asset)
    return consent


def new_run_dir(runs: Path, prefix: str, now: datetime | None = None) -> Path:
    """A new, empty run folder named <prefix>-<YYYYMMDD-HHMMSS>, created at once.

    Creating it reserves the name, so two runs started in the same second
    never share (and overwrite) a folder: the later one takes the next free
    second.
    """
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    when = now or datetime.now()
    for _ in range(3600):
        folder = runs / f"{prefix}-{when:%Y%m%d-%H%M%S}"
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            when += timedelta(seconds=1)
    raise RuntimeError(f"no free run folder name for {prefix} in {runs}")


def list_assets(library: Path | None = None) -> list[Asset]:
    library = library or default_library()
    if not library.is_dir():
        return []
    assets = [
        load_asset(entry)
        for entry in sorted(library.iterdir())
        if entry.is_dir() and not entry.name.startswith(".") and (entry / MANIFEST).is_file()
    ]
    return sorted(assets, key=lambda asset: asset.ingested_at)


def _write_manifest(asset: Asset) -> None:
    temp = asset.directory / f"{MANIFEST}.tmp"
    temp.write_text(json.dumps(asset.to_manifest(), indent=2) + "\n")
    os.replace(temp, asset.directory / MANIFEST)


def check_decodable(path: Path) -> None:
    """Decode the first frame, so a codec this machine lacks fails now, not mid-analysis."""
    result = subprocess.run(
        [
            require_tool("ffmpeg"), "-v", "error", "-nostdin",
            "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise IngestError(f"Can't decode {path.name} on this machine: {detail}")


def make_proxy(asset: Asset, max_side: int = DEFAULT_MAX_SIDE) -> dict:
    """Write a small constant-frame-rate H.264 copy next to the source.

    Video only. Decoding it is much cheaper than a 4K HEVC original when
    detection and tracking experiments run over the same game many times.
    """
    video = asset.video
    width, height = fit_within(video.display_width, video.display_height, max_side)
    temp = asset.directory / f".{PROXY}.part"
    result = subprocess.run(
        [
            require_tool("ffmpeg"), "-v", "error", "-nostdin", "-y",
            "-i", str(asset.source_path),
            "-map", "0:v:0",
            "-vf", f"fps={video.avg_frame_rate},scale={width}:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            "-movflags", "+faststart",
            "-f", "mp4",
            str(temp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        temp.unlink(missing_ok=True)
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise IngestError(f"Proxy encode failed for {asset.asset_id}: {detail}")
    os.replace(temp, asset.directory / PROXY)
    asset.proxy = {
        "path": PROXY,
        "width": width,
        "height": height,
        "avg_frame_rate": video.avg_frame_rate,
        "max_side": max_side,
    }
    _write_manifest(asset)
    return asset.proxy


def ingest(
    path: str | Path,
    library: Path | None = None,
    *,
    link: bool = False,
    move: bool = False,
    proxy: bool = False,
    proxy_max_side: int = DEFAULT_MAX_SIDE,
    original_filename: str | None = None,
) -> tuple[Asset, bool]:
    """Add a video to the library. Returns (asset, created).

    Validates before copying anything. Ingesting a file that is already in
    the library returns the existing asset with created=False (adding the
    proxy if it was asked for and is missing). The asset folder appears only
    once complete, so an interrupted ingest leaves nothing half-written.

    `link` symlinks the original instead of copying it. `move` moves it into
    the library (instant on the same disk, used for uploads); if the ingest
    fails or the file turns out to be a duplicate, it is left where it was.
    `original_filename` overrides the name recorded in the manifest.
    """
    if link and move:
        raise ValueError("link and move can't both be used")
    path = Path(path)
    library = library or default_library()
    try:
        info = probe(path)
    except MediaError as error:
        raise IngestError(str(error)) from error
    check_decodable(path)

    digest = file_sha256(path)
    asset_id = digest[:ASSET_ID_LENGTH]
    final = library / asset_id

    if (final / MANIFEST).is_file():
        asset = load_asset(final)
        if proxy and asset.proxy is None:
            make_proxy(asset, proxy_max_side)
        return asset, False

    library.mkdir(parents=True, exist_ok=True)
    staging = library / f".ingest-{asset_id}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    source = "source" + path.suffix.lower()

    def put_back() -> None:
        if move and (staging / source).exists() and not path.exists():
            shutil.move(staging / source, path)

    try:
        if link:
            (staging / source).symlink_to(path.resolve())
        elif move:
            shutil.move(path, staging / source)
        else:
            shutil.copy2(path, staging / source)
        asset = Asset(
            directory=staging,
            asset_id=asset_id,
            sha256=digest,
            original_filename=original_filename or path.name,
            ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source=source,
            source_mode="symlink" if link else "move" if move else "copy",
            video=info,
        )
        _write_manifest(asset)
        if proxy:
            make_proxy(asset, proxy_max_side)
        try:
            staging.rename(final)
        except OSError:
            # Another ingest of the same file finished first: keep theirs.
            if (final / MANIFEST).is_file():
                put_back()
                shutil.rmtree(staging, ignore_errors=True)
                return load_asset(final), False
            raise
    except BaseException:
        put_back()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    asset.directory = final
    return asset, True
