"""Command line: soccer-recorder doctor | probe | ingest | list | make-sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from .frames import DEFAULT_MAX_SIDE
from .library import IngestError, default_library, ingest, list_assets
from .media import MediaError, VideoInfo, probe
from .sample import make_sample

MIN_PYTHON = (3, 12)


def _duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{secs:06.3f}" if hours else f"{minutes}:{secs:06.3f}"


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _describe(info: VideoInfo) -> list[str]:
    lines = [
        f"  file      {info.filename} ({_size(info.size_bytes)}, {info.container})",
        f"  duration  {_duration(info.duration_s)} ({info.duration_s:.3f} s)",
        f"  video     {info.codec} {info.display_width}x{info.display_height}"
        f" @ {info.fps:.3f} fps{' (VFR)' if info.is_vfr else ''}"
        f"{f', rotated {info.rotation}°' if info.rotation else ''}"
        f"{', HDR' if info.is_hdr else ''}",
        f"  audio     {'yes' if info.has_audio else 'no'}",
    ]
    if info.device:
        lines.append(f"  device    {info.device}")
    if info.creation_time:
        lines.append(f"  recorded  {info.creation_time}")
    lines += [f"  warning   {warning}" for warning in info.warnings]
    return lines


def cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.video)
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print("\n".join(_describe(info)))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    library = args.library or default_library()
    asset, created = ingest(
        args.video,
        library,
        link=args.link,
        proxy=args.proxy,
        proxy_max_side=args.proxy_max_side,
    )
    if args.json:
        print(json.dumps({"created": created, **asset.to_manifest()}, indent=2))
        return 0
    verb = "Ingested" if created else "Already in library"
    print(f"{verb}: {asset.asset_id}  ({asset.directory})")
    print("\n".join(_describe(asset.video)))
    if asset.proxy:
        print(f"  proxy     {asset.proxy['width']}x{asset.proxy['height']}  {asset.proxy_path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    library = args.library or default_library()
    assets = list_assets(library)
    if args.json:
        print(json.dumps([asset.to_manifest() for asset in assets], indent=2))
        return 0
    if not assets:
        print(f"No videos in {library}. Add one with: soccer-recorder ingest <video>")
        return 0
    for asset in assets:
        video = asset.video
        print(
            f"{asset.asset_id}  {_duration(video.duration_s):>12}  "
            f"{video.display_width}x{video.display_height}  "
            f"{'proxy ' if asset.proxy else '      '}{asset.original_filename}"
        )
    total = sum(asset.video.duration_s for asset in assets)
    print(f"{len(assets)} video(s), {total / 60:.1f} minutes in {library}")
    return 0


def cmd_make_sample(args: argparse.Namespace) -> int:
    path = make_sample(
        args.out,
        duration=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        codec=args.codec,
        rotation=args.rotation,
    )
    print(f"Wrote synthetic sample {path}")
    return 0


def _tool_version(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    result = subprocess.run([path, "-version"], capture_output=True, text=True)
    first = result.stdout.splitlines()[0] if result.stdout else ""
    return first.split(" Copyright")[0] or path


def _ffmpeg_list(flag: str) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ""
    return subprocess.run([ffmpeg, "-hide_banner", flag], capture_output=True, text=True).stdout


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check this machine can run the pipeline. Exit 1 if anything required is missing."""
    checks: list[tuple[str, str, str]] = []

    def check(status: str, name: str, detail: str) -> None:
        checks.append((status, name, detail))

    version = ".".join(map(str, sys.version_info[:3]))
    check("ok" if sys.version_info >= MIN_PYTHON else "FAIL", "python", version)

    for tool in ("ffmpeg", "ffprobe"):
        found = _tool_version(tool)
        check("ok" if found else "FAIL", tool, found or "not found: brew install ffmpeg")

    decoders, encoders = _ffmpeg_list("-decoders"), _ffmpeg_list("-encoders")
    if decoders:
        for codec in ("h264", "hevc"):
            ok = f" {codec} " in decoders
            check("ok" if ok else "FAIL", f"decode {codec}", "available" if ok else "missing")
        ok = "libx264" in encoders
        check("ok" if ok else "warn", "encode libx264", "available" if ok else
              "missing: proxies and exports need an FFmpeg built with libx264")

    numpy_spec = importlib.util.find_spec("numpy")
    check("ok" if numpy_spec else "FAIL", "numpy", "installed" if numpy_spec else "run uv sync")

    library = args.library or default_library()
    try:
        library.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=library):
            pass
        check("ok", "library", f"{library.resolve()} (writable)")
    except OSError as error:
        check("FAIL", "library", f"{library}: {error}")

    if importlib.util.find_spec("torch"):
        import torch

        mps = "available" if torch.backends.mps.is_available() else "unavailable"
        check("ok", "torch", f"{torch.__version__}, Apple MPS {mps}")
    else:
        check("info", "torch", "not installed (not needed until R2 detection)")

    for status, name, detail in checks:
        print(f"[{status:>4}] {name:<15} {detail}")
    failed = [name for status, name, _ in checks if status == "FAIL"]
    print("All required checks passed." if not failed else f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccer-recorder",
        description="AI Soccer Recorder proof of concept (R1: ingestion).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    library_help = (
        "library folder (default: $SOCCER_RECORDER_LIBRARY or ./data/library)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check Python, FFmpeg and codecs are set up")
    doctor.add_argument("--library", type=Path, help=library_help)
    doctor.set_defaults(func=cmd_doctor)

    probe_cmd = sub.add_parser("probe", help="show a video's metadata without ingesting it")
    probe_cmd.add_argument("video", type=Path)
    probe_cmd.add_argument("--json", action="store_true", help="print JSON")
    probe_cmd.set_defaults(func=cmd_probe)

    ingest_cmd = sub.add_parser("ingest", help="validate a video and add it to the library")
    ingest_cmd.add_argument("video", type=Path)
    ingest_cmd.add_argument("--library", type=Path, help=library_help)
    ingest_cmd.add_argument(
        "--link", action="store_true",
        help="symlink the original instead of copying it (saves disk; don't move the original)",
    )
    ingest_cmd.add_argument(
        "--proxy", action="store_true",
        help="also write a small constant-frame-rate proxy.mp4 for detection",
    )
    ingest_cmd.add_argument(
        "--proxy-max-side", type=int, default=DEFAULT_MAX_SIDE,
        help=f"longest side of the proxy in pixels (default {DEFAULT_MAX_SIDE})",
    )
    ingest_cmd.add_argument("--json", action="store_true", help="print the manifest as JSON")
    ingest_cmd.set_defaults(func=cmd_ingest)

    list_cmd = sub.add_parser("list", help="list videos in the library")
    list_cmd.add_argument("--library", type=Path, help=library_help)
    list_cmd.add_argument("--json", action="store_true", help="print JSON")
    list_cmd.set_defaults(func=cmd_list)

    sample = sub.add_parser("make-sample", help="write a synthetic test clip (no real people)")
    sample.add_argument("out", type=Path)
    sample.add_argument("--duration", type=float, default=10.0, help="seconds (default 10)")
    sample.add_argument("--width", type=int, default=1920)
    sample.add_argument("--height", type=int, default=1080)
    sample.add_argument("--fps", type=float, default=30)
    sample.add_argument("--codec", choices=["h264", "hevc"], default="hevc",
                        help="hevc matches what iPhones record (default)")
    sample.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270],
                        help="display rotation metadata, as in a portrait recording")
    sample.set_defaults(func=cmd_make_sample)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (MediaError, IngestError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode(errors="replace").strip()
        print(f"error: {' '.join(map(str, error.cmd[:1]))} failed: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
