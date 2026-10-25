"""Command line: soccer-recorder doctor | probe | ingest | list | make-sample |
detect | benchmark | sample-frames | evaluate | track | follow | analyze | serve."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .detect import Detector, Progress, run_detection
from .evaluate import LabelError, detect_images, load_coco, score
from .frames import DEFAULT_MAX_SIDE, sample_frames
from .library import (
    USES,
    Asset,
    IngestError,
    default_library,
    ingest,
    list_assets,
    new_run_dir,
    set_consent,
)
from .media import MediaError, VideoInfo, probe
from .rfdetr_detector import (
    DEFAULT_SIZE,
    DEFAULT_THRESHOLD,
    SIZES,
    DetectorUnavailable,
    RFDETRDetector,
    parse_shape,
)
from .sample import make_sample
from .tracking import load_or_track

MIN_PYTHON = (3, 12)


def _duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{secs:06.3f}" if hours else f"{minutes}:{secs:06.3f}"


def _eta(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


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


USE_TEXT = {
    "testing": "measuring accuracy (labelled frames, reviews)",
    "training": "training our own models",
    "public": "showing it publicly (e.g. the campaign video)",
}


def cmd_consent(args: argparse.Namespace) -> int:
    _, asset = _library_asset(args.target, args.library, "consent")
    changes = {use: getattr(args, use) for use in USES if getattr(args, use) is not None}
    if changes or args.note is not None:
        set_consent(asset, note=args.note, **changes)
    print(f"{asset.asset_id}  {asset.original_filename}")
    for use in USES:
        print(f"  {'yes' if asset.allows(use) else 'no ':<4} {USE_TEXT[use]}")
    consent = asset.consent or {}
    if consent.get("note"):
        print(f"  note  {consent['note']}")
    if consent.get("updated_at"):
        print(f"  set   {consent['updated_at']}")
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

    if importlib.util.find_spec("rfdetr") and importlib.util.find_spec("torch"):
        import torch

        mps = "available" if torch.backends.mps.is_available() else "unavailable"
        cuda = "available" if torch.cuda.is_available() else "unavailable"
        check("ok", "detect extra", f"rfdetr {importlib.metadata.version('rfdetr')}, "
              f"torch {torch.__version__}; Apple MPS {mps}, CUDA {cuda}")
    else:
        check("info", "detect extra",
              "not installed: uv sync --extra detect (needed for detect, benchmark, evaluate)")

    for status, name, detail in checks:
        print(f"[{status:>4}] {name:<15} {detail}")
    failed = [name for status, name, _ in checks if status == "FAIL"]
    print("All required checks passed." if not failed else f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


def make_detector(
    size: str = DEFAULT_SIZE,
    *,
    device: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    shape: tuple[int, int] | None = None,
    weights: Path | None = None,
) -> Detector:
    """The detector the commands use. Tests replace this with a stand-in."""
    return RFDETRDetector(size, device=device, threshold=threshold, shape=shape, weights=weights)


def _weights(args: argparse.Namespace) -> dict:
    """`weights=` for make_detector, only when a fine-tuned model was asked for."""
    weights = getattr(args, "weights", None)
    return {"weights": weights} if weights else {}


def _resolve(target: str, library: Path) -> tuple[Path, Asset | None]:
    """A video path, or a library asset id (or a unique prefix of one, 4+ characters)."""
    path = Path(target)
    if path.is_file():
        return path, None
    matches = (
        [asset for asset in list_assets(library) if asset.asset_id.startswith(target)]
        if len(target) >= 4 else []
    )
    if len(matches) == 1:
        return matches[0].source_path, matches[0]
    if matches:
        raise ValueError(f"{target!r} matches {len(matches)} assets; give more characters")
    raise ValueError(f"{target}: no such file, and no asset in {library} starts with that id")


def _runs_base(video: Path, asset: Asset | None) -> Path:
    return asset.directory / "runs" if asset else Path("data/runs") / video.stem


def _run_dir(video: Path, asset: Asset | None, name: str) -> Path:
    """A new run folder, created now (see library.new_run_dir)."""
    return new_run_dir(_runs_base(video, asset), name)


@contextlib.contextmanager
def _removed_on_error(folder: Path, made: bool):
    """Remove a run folder this command made if the command fails, so no half run is left."""
    try:
        yield
    except BaseException:
        if made:
            shutil.rmtree(folder, ignore_errors=True)
        raise


def _progress_printer():
    last = [0.0]

    def show(progress: Progress) -> None:
        now = time.monotonic()
        done = progress.video_s >= progress.total_video_s - 0.5
        if now - last[0] < 2 and not done:
            return
        last[0] = now
        total = progress.total_video_s
        percent = 100 * progress.video_s / total if total else 100.0
        rate = progress.video_s / progress.elapsed_s if progress.elapsed_s else 0.0
        eta = (total - progress.video_s) / rate if rate else 0.0
        fps = progress.frames / progress.elapsed_s if progress.elapsed_s else 0.0
        print(
            f"\r  {progress.video_s:7.1f} / {total:.1f} s of video ({percent:5.1f}%)"
            f"  {fps:5.1f} frames/s  ETA {_eta(eta)}   ",
            end="", file=sys.stderr, flush=True,
        )

    return show


def _print_metrics(metrics: dict, out: Path) -> None:
    speed = metrics["speed"]
    people = metrics["detections"].get("person", {})
    ball = metrics["detections"].get("sports ball", {})
    heights = metrics.get("person_height_px") or {}
    print(f"  frames        {metrics['frames']} ({metrics['window']['seconds']:.1f} s of video)")
    p50 = (speed["detect_ms"] or {}).get("p50")
    print(f"  speed         {speed['realtime_factor']}× real time "
          f"({speed['frames_per_s']} frames/s; detect p50 {p50} ms)")
    if people:
        print(f"  players       {people['per_frame']} per frame, mean score {people['mean_score']}")
    if heights:
        print(f"  player height p10 {heights['p10']} / p50 {heights['p50']} "
              f"/ p90 {heights['p90']} px")
    if ball:
        print(f"  ball          seen in {100 * ball['frames_with_any']:.0f}% of frames")
    print(f"  outputs       {out}")


def cmd_detect(args: argparse.Namespace) -> int:
    video, asset = _resolve(args.target, args.library or default_library())
    detector = make_detector(
        args.size, device=args.device, threshold=args.threshold, shape=parse_shape(args.shape),
        **_weights(args),
    )
    out = args.out or _run_dir(video, asset, f"detect-{detector.name}")
    print(f"Detecting with {detector.name} on {detector.describe()['device']}: {video.name}")
    with _removed_on_error(out, made=not args.out):
        metrics = run_detection(
            video, detector, out,
            max_side=args.max_side, fps=args.fps, start_s=args.start, duration_s=args.duration,
            write_video=not args.no_video, on_progress=_progress_printer(),
        )
    print(file=sys.stderr)
    _print_metrics(metrics, out)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Time several detector sizes and input shapes on the same stretch of video."""
    video, asset = _resolve(args.target, args.library or default_library())
    sizes = [size.strip() for size in args.sizes.split(",") if size.strip()]
    shapes: list[tuple[int, int] | None] = [None]
    if args.shapes:
        shapes += [parse_shape(text) for text in args.shapes.split(",")]
    base = args.out or _run_dir(video, asset, "benchmark")
    rows = []
    for size in sizes:
        for shape in shapes:
            detector = make_detector(size, device=args.device, threshold=args.threshold,
                                     shape=shape)
            described = detector.describe()
            label = "x".join(map(str, described.get("input_shape", []))) or "default"
            # Warm up once so one-off start-up costs don't count against speed.
            detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))
            print(f"{detector.name} at {label} on {described['device']}", file=sys.stderr)
            metrics = run_detection(
                video, detector, base / f"{detector.name}-{label}",
                max_side=args.max_side, fps=args.fps, start_s=args.start,
                duration_s=args.duration, write_video=False, on_progress=_progress_printer(),
            )
            print(file=sys.stderr)
            rows.append(metrics)

    print("| Detector | Input | Device | × real time | Detect p50 ms | Players/frame "
          "| Player height p50 px | Ball seen |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for metrics in rows:
        d, speed = metrics["detector"], metrics["speed"]
        people = metrics["detections"].get("person", {})
        ball = metrics["detections"].get("sports ball", {})
        print(
            f"| {d['name']} | {'x'.join(map(str, d.get('input_shape', [])))} | {d['device']} "
            f"| {speed['realtime_factor']} | {(speed['detect_ms'] or {}).get('p50')} "
            f"| {people.get('per_frame')} | {(metrics['person_height_px'] or {}).get('p50')} "
            f"| {100 * ball.get('frames_with_any', 0):.0f}% |"
        )
    (base / "benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nDetails: {base / 'benchmark.json'}")
    return 0


def cmd_sample_frames(args: argparse.Namespace) -> int:
    video, asset = _resolve(args.target, args.library or default_library())
    prefix = asset.asset_id if asset else video.stem
    out = args.out or Path("data/labeling") / prefix
    frames = sample_frames(video, out, every_s=args.every, prefix=prefix,
                           start_s=args.start, duration_s=args.duration)
    print(f"Wrote {len(frames)} frames to {out}")
    print("Label them in CVAT (docs/labeling.md), export as COCO 1.0, then run:")
    print(f"  soccer-recorder evaluate --labels <export>.json --images {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    labels = load_coco(args.labels)
    if not labels:
        raise LabelError(f"{args.labels}: no images")
    detector = make_detector(args.size, device=args.device, threshold=args.min_score,
                             shape=parse_shape(args.shape), **_weights(args))

    def on_image(n: int, total: int) -> None:
        print(f"\r  {n}/{total} images", end="", file=sys.stderr, flush=True)

    detections = detect_images(detector, labels, args.images, max_side=args.max_side,
                               on_image=on_image)
    print(file=sys.stderr)
    report = score(detections, labels, threshold=args.threshold)
    report["detector"] = detector.describe()
    report["max_side"] = args.max_side
    report["labels_file"] = Path(args.labels).name

    print(f"{report['images']} images, IoU >= {report['iou_threshold']}, "
          f"score >= {report['score_threshold']}, {detector.name}")
    print("| Class | Labelled | Precision | Recall | F1 | AP50 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, result in report["classes"].items():
        print(f"| {name} | {result['labelled']} | {result['precision']} | {result['recall']} "
              f"| {result['f1']} | {result['ap50']} |")
    for name, result in report["classes"].items():
        buckets = ", ".join(
            f"{bucket}px {b['found']}/{b['labelled']}"
            for bucket, b in result["recall_by_height_px"].items() if b["labelled"]
        )
        if buckets:
            print(f"{name} recall by height: {buckets}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or Path(args.labels).with_name(f"eval-{detector.name}-{stamp}.json")
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {out}")
    return 0


def _parse_point(text: str, info: VideoInfo) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"--point must look like 640,360 (or 0.5,0.5), not {text!r}") from None
    if 0 <= x <= 1 and 0 <= y <= 1:  # fractions of the frame
        return x * info.display_width, y * info.display_height
    return x, y


def cmd_track(args: argparse.Namespace) -> int:
    from .overlay import render_tracks

    video, asset = _resolve(args.target, args.library or default_library())
    detector = make_detector(args.size, device=args.device, threshold=args.threshold,
                             **_weights(args))
    print(f"Tracking players with {detector.name} on {detector.describe()['device']}: "
          f"{video.name}")
    tracks, folder, cached = load_or_track(
        video, _runs_base(video, asset), detector, fps=args.fps, start_s=args.start,
        duration_s=args.duration, max_side=args.max_side, fresh=args.fresh,
        on_progress=_progress_printer())
    print(file=sys.stderr)
    metrics = json.loads((folder / "metrics.json").read_text())
    if not args.no_video:
        render_tracks(video, tracks, folder / "tracks.mp4", max_side=args.max_side)
    print(f"  {'reused' if cached else 'tracked'}     {metrics['frames']} frames, "
          f"{metrics['tracks']['count']} tracks (median {metrics['tracks']['median_length_s']} s), "
          f"{metrics['tracks']['per_frame']} players per frame")
    print(f"  outputs     {folder}")
    return 0


def cmd_follow(args: argparse.Namespace) -> int:
    from .follow import Seed, Target, follow_player

    video, asset = _resolve(args.target, args.library or default_library())
    info = probe(video)
    x, y = _parse_point(args.point, info)
    target = Target(number=args.number, name=args.name or "")
    detector = make_detector(args.size, device=args.device, threshold=args.threshold,
                             **_weights(args))
    print(f"Following {target.label} from {args.at:g} s in {video.name}")
    tracks, folder, cached = load_or_track(
        video, _runs_base(video, asset), detector, fps=args.fps, start_s=args.start,
        duration_s=args.duration, max_side=args.max_side, on_progress=_progress_printer())
    print(file=sys.stderr)
    out = args.out or _run_dir(video, asset, f"follow-n{args.number}")
    with _removed_on_error(out, made=not args.out):
        metrics = follow_player(video, tracks, Seed(args.at, x, y), target, out,
                                max_side=args.max_side, on_progress=_progress_printer(),
                                tracks_run=folder.name)
    print(file=sys.stderr)
    from .analytics import analyse, load_goals

    result = analyse(video, tracks, out,
                     goals=load_goals(asset.directory, tracks.size) if asset else [],
                     player_height_m=args.height_cm / 100 if args.height_cm else None)
    coverage = metrics["coverage"]
    print(f"  found       {100 * (coverage['seen'] + coverage['filled']):.0f}% of the time "
          f"({100 * coverage['seen']:.0f}% seen, {100 * coverage['filled']:.0f}% filled in)")
    print(f"  links       {metrics['links']} ({metrics['uncertain_links']} uncertain)")
    _print_analytics(result)
    print(f"  tracks      {'reused from' if cached else 'saved in'} {folder.name}")
    print(f"  outputs     {out}")
    return 0


def _print_analytics(result: dict) -> None:
    stats = result["stats"]
    print(f"  on the ball {stats['on_ball_s']:.0f} s, {stats['touches']} touches, "
          f"{stats['dribbles']} dribbles")
    print(f"  passes      {stats['passes_completed']}/{stats['passes_made']} completed, "
          f"{stats['passes_received']} received; {stats['shots']} shots")
    print(f"  running     ~{stats['distance_m']:.0f} m, top speed ~{stats['top_speed_kmh']} km/h, "
          f"{stats['sprints']} sprints")
    goals = [e for e in result["events"] if e["type"] in ("goal", "possible_goal")]
    for goal in goals:
        kind = "goal" if goal["type"] == "goal" else "possible goal"
        print(f"  {kind:<11} at {goal['t']:.1f} s ({100 * goal['confidence']:.0f}%: "
              f"{', '.join(goal['details'].get('evidence', []))})")
    for note in result["notes"]:
        print(f"  note        {note}")


def _parse_goals(text: str) -> list[list[float]]:
    try:
        return [[float(v) for v in part.split(",")] for part in text.split(";") if part.strip()]
    except ValueError:
        raise ValueError("--goals looks like 0,0.4,0.05,0.7;0.95,0.4,1,0.7 "
                         "(x1,y1,x2,y2 as fractions of the frame, one box per goal)") from None


def _follow_run(asset: Asset, name: str | None) -> Path:
    """A follow run folder by name, or the newest one."""
    runs = sorted((p for p in (asset.directory / "runs").glob("follow-*")
                   if (p / "target.json").is_file()), key=lambda p: p.name[-15:])  # by time
    run = asset.directory / "runs" / name if name else (runs[-1] if runs else None)
    if run is None or not (run / "target.json").is_file():
        raise ValueError("no follow run" + (f" {name!r}" if name else "")
                         + ": run `soccer-recorder follow` first")
    return run


def _library_asset(target: str, library: Path | None, command: str) -> tuple[Path, Asset]:
    video, asset = _resolve(target, library or default_library())
    if asset is None:
        raise ValueError(f"{command} works on library videos: ingest the video first")
    return video, asset


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analytics import analyse_run, save_goals

    video, asset = _library_asset(args.target, args.library, "analyze")
    if args.goals is not None:
        save_goals(asset.directory, _parse_goals(args.goals))
    run = _follow_run(asset, args.run)
    result = analyse_run(video, run, goals_dir=asset.directory,
                         player_height_m=args.height_cm / 100 if args.height_cm else None)
    print(f"Analytics for {result['target']['name'] or ''} #{result['target']['number']} "
          f"({run.name})")
    _print_analytics(result)
    print(f"  saved       {run / 'analytics.json'}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .exports import make_export

    video, asset = _library_asset(args.target, args.library, "export")
    run = _follow_run(asset, args.run)
    what = {"playercam": "player cam", "highlights": "highlight reel"}[args.kind]
    print(f"Making the {what} ({'vertical' if args.vertical else 'landscape'}) from {run.name}")
    made = make_export(video, run, args.kind, vertical=args.vertical,
                       on_progress=_progress_printer())
    print(file=sys.stderr)
    if made.get("moments"):
        for start, end, caption in made["moments"]:
            print(f"  moment      {start:.1f}-{end:.1f} s  {caption}")
    print(f"  length      {made['seconds']:.1f} s")
    print(f"  saved       {run / made['file']}")
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    from .analytics import load_trajectory
    from .exports import correct_run
    from .follow import Seed

    video, asset = _library_asset(args.target, args.library, "fix")
    run = _follow_run(asset, args.run)
    _, saved = load_trajectory(run / "target.json")
    x, y = _parse_point(args.point, probe(video))
    out = _run_dir(video, asset, f"follow-n{saved['target']['number']}")
    print(f"Following #{saved['target']['number']} again from {run.name}, "
          f"with a tap at {args.at:g} s")
    with _removed_on_error(out, made=True):
        metrics = correct_run(video, run, Seed(args.at, x, y), out, goals_dir=asset.directory,
                              render=not args.no_video, on_progress=_progress_printer())
    print(file=sys.stderr)
    coverage = metrics["coverage"]
    print(f"  taps        {len(metrics['taps'])} (the first, plus {len(metrics['taps']) - 1} "
          f"fix{'es' if len(metrics['taps']) > 2 else ''})")
    print(f"  found       {100 * (coverage['seen'] + coverage['filled']):.0f}% of the time")
    print(f"  reviews     {metrics['reviews_kept']} kept")
    _print_analytics(json.loads((out / "analytics.json").read_text()))
    print(f"  outputs     {out}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import build_dataset

    labels, images = args.labels or [], args.images or []
    if not labels or len(labels) != len(images):
        raise ValueError("give --labels and --images in pairs: each labels file with the "
                         "folder of the frames it labels")
    out = args.out or Path("data/datasets") / args.name
    sheet = build_dataset(list(zip(labels, images, strict=True)), out,
                          library=args.library or default_library(), seed=args.seed)
    print(f"Training set {sheet['name']}: {len(sheet['videos'])} video(s), split by "
          f"{sheet['split_by']}")
    for split, counts in sheet["splits"].items():
        print(f"  {split:<6} {counts['images']:>5} frames  {counts['person']:>6} people  "
              f"{counts['sports ball']:>5} balls")
    refused = sheet["refused"]
    no_permission = sum(r["frames"] for r in refused["no training permission"])
    if no_permission or refused["not from a library video"] or refused["image file missing"]:
        print(f"  left out {no_permission} frames without training permission, "
              f"{refused['not from a library video']} not from a library video, "
              f"{refused['image file missing']} with the image missing")
    print(f"  saved  {out} (datasheet.md says where every frame came from)")
    return 0


def cmd_train_detector(args: argparse.Namespace) -> int:
    from .finetune import train_detector

    out = args.out or Path("data/models") / args.name
    card = train_detector(
        args.dataset, out, size=args.size, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, device=args.device,
        resolution=args.resolution, workers=args.workers, max_side=args.max_side,
        on_stage=lambda message: print(message, file=sys.stderr),
    )
    test = card["test"]
    print(f"Model {card['name']} (RF-DETR {card['size']}, {card['training']['epochs']} epochs "
          f"in {card['training']['wall_s'] / 60:.1f} min on {card['training']['device']})")
    print(f"  test frames {test['frames']} (from {card['dataset']['split_by']}s it didn't "
          "train on)")
    for label in ("person", "sports ball"):
        before, after = test["before"][label], test["after"][label]

        def show(r):
            return " ".join(f"{k} {'-' if r[k] is None else f'{r[k]:.3f}'}"
                            for k in ("precision", "recall", "ap50"))

        print(f"  {label:<12} before: {show(before)}")
        print(f"  {'':<12} after:  {show(after)}")
    if test["better"]:
        print(f"  Better on the test frames. Use it with: --weights {out}")
    else:
        print("  Not better than the standard model: keep using that. Label more frames, "
              "or train longer.")
    return 0


def cmd_reviews(args: argparse.Namespace) -> int:
    from .calibration import MIN_REVIEWED, ONLY_TICKED, update_calibration

    library = args.library or default_library()
    calibration = update_calibration(library)
    stats = calibration["stats"]
    if not stats["checked"]:
        print("No moments checked yet. Tick ✓ or cross ✗ them on the Analytics page.")
        return 0
    print(f"Your checks: {stats['checked']} moments in {stats['games']} game(s)")
    print(f"  {'moment':<16} {'checked':>7} {'right':>6} {'right %':>8}  highlight reel takes")
    for kind, s in stats["types"].items():
        threshold = s["min_confidence"]
        if threshold is None:
            uses = f"≥ 60% sure (default; learns after {MIN_REVIEWED - s['checked']} more)"
        elif threshold >= ONLY_TICKED:
            uses = "only the ones you tick (too often wrong)"
        else:
            uses = f"≥ {100 * threshold:.0f}% sure (learnt)"
        print(f"  {kind:<16} {s['checked']:>7} {s['right']:>6} {100 * s['precision']:>7.0f}%  "
              f"{uses}")
    print(f"  saved  {Path(library) / 'calibration.json'}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import App, make_server

    library = args.library or default_library()
    app = App(library, detector_factory=lambda size: make_detector(size, **_weights(args)))
    try:
        server = make_server(app, args.port)
    except OSError as error:
        raise ValueError(f"can't use port {args.port} ({error.strerror}); "
                         f"try --port {args.port + 1}") from error
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Soccer Recorder is running at {url}")
    print(f"Library: {library.resolve()}")
    print("Only this computer can open it. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping (cancelling any running jobs)…")
    finally:
        server.server_close()
        app.jobs.shutdown()
    return 0


def _add_detector_args(parser: argparse.ArgumentParser, *, sizes: bool = False) -> None:
    if not sizes:
        parser.add_argument("--size", choices=list(SIZES), default=DEFAULT_SIZE,
                            help=f"RF-DETR size (default {DEFAULT_SIZE}); only "
                                 "the Apache-2.0 sizes are offered")
    parser.add_argument("--device", default="auto",
                        help="auto (Apple GPU, then CUDA, then CPU), mps, cuda or cpu")
    if not sizes:
        parser.add_argument("--weights", type=Path,
                            help="a fine-tuned model folder from train-detector (sets the size)")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"downscale frames so the long side is at most this "
                             f"(default {DEFAULT_MAX_SIDE})")


def _add_window_args(parser: argparse.ArgumentParser, duration: float | None = None) -> None:
    parser.add_argument("--start", type=float, default=0.0, help="start at this second")
    parser.add_argument("--duration", type=float, default=duration,
                        help="seconds of video to use" + (f" (default {duration:g})"
                                                          if duration else " (default: all)"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccer-recorder",
        description="AI Soccer Recorder proof of concept: ingest, detect, track, follow, "
                    "analyse, and make player cams and highlight reels.",
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

    consent = sub.add_parser(
        "consent", help="record what the families agreed a video may be used for",
        description="Every family whose child can be identified must agree. A new video "
                    "has no permissions: it's only for watching on this computer.")
    consent.add_argument("target", help="a library asset id (or its start)")
    for use in USES:
        consent.add_argument(f"--{use}", action=argparse.BooleanOptionalAction, default=None,
                             help=USE_TEXT[use])
    consent.add_argument("--note", help="where the signed permission is kept (no children's "
                                        "names)")
    consent.add_argument("--library", type=Path, help=library_help)
    consent.set_defaults(func=cmd_consent)

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

    detect = sub.add_parser("detect", help="find players and the ball through a video")
    detect.add_argument("target", help="a video file, or a library asset id (or its start)")
    _add_detector_args(detect)
    detect.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"minimum detection score (default {DEFAULT_THRESHOLD})")
    detect.add_argument("--shape", help="model input as WIDTHxHEIGHT, e.g. 704x384 for wide "
                                        "frames (default: the model's square size)")
    detect.add_argument("--fps", type=float, help="analyse this many frames per second "
                                                  "(default: the video's rate)")
    _add_window_args(detect)
    detect.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
    detect.add_argument("--library", type=Path, help=library_help)
    detect.add_argument("--out", type=Path, help="run folder (default: under the asset's runs/)")
    detect.set_defaults(func=cmd_detect)

    bench = sub.add_parser("benchmark", help="time detector sizes and input shapes on a clip")
    bench.add_argument("target", help="a video file, or a library asset id (or its start)")
    bench.add_argument("--sizes", default="nano,small,medium",
                       help="comma-separated RF-DETR sizes (default nano,small,medium)")
    bench.add_argument("--shapes", help="extra input shapes to try, e.g. 704x384,960x544")
    _add_detector_args(bench, sizes=True)
    bench.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    bench.add_argument("--fps", type=float)
    _add_window_args(bench, duration=60.0)
    bench.add_argument("--library", type=Path, help=library_help)
    bench.add_argument("--out", type=Path)
    bench.set_defaults(func=cmd_benchmark)

    sampler = sub.add_parser("sample-frames", help="save frames to label for evaluation")
    sampler.add_argument("target", help="a video file, or a library asset id (or its start)")
    sampler.add_argument("--every", type=float, default=10.0,
                         help="seconds between frames (default 10)")
    _add_window_args(sampler)
    sampler.add_argument("--library", type=Path, help=library_help)
    sampler.add_argument("--out", type=Path,
                         help="folder (default data/labeling/<asset id or file name>)")
    sampler.set_defaults(func=cmd_sample_frames)

    evaluate = sub.add_parser("evaluate", help="score detection against labelled frames")
    evaluate.add_argument("--labels", type=Path, required=True,
                          help="COCO JSON exported from CVAT")
    evaluate.add_argument("--images", type=Path, required=True,
                          help="folder with the labelled images")
    _add_detector_args(evaluate)
    evaluate.add_argument("--shape", help="model input as WIDTHxHEIGHT")
    evaluate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                          help=f"score cut-off for precision/recall (default {DEFAULT_THRESHOLD})")
    evaluate.add_argument("--min-score", type=float, default=0.05,
                          help="lowest score kept, for AP50 (default 0.05)")
    evaluate.add_argument("--out", type=Path, help="report JSON (default next to the labels)")
    evaluate.set_defaults(func=cmd_evaluate)

    track = sub.add_parser("track", help="track every player; draws each track's id (R3)")
    track.add_argument("target", help="a video file, or a library asset id (or its start)")
    _add_detector_args(track)
    track.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    track.add_argument("--fps", type=float, default=15.0,
                       help="frames analysed per second (default 15)")
    _add_window_args(track)
    track.add_argument("--fresh", action="store_true", help="don't reuse earlier tracks")
    track.add_argument("--no-video", action="store_true", help="skip tracks.mp4")
    track.add_argument("--library", type=Path, help=library_help)
    track.set_defaults(func=cmd_track)

    follow = sub.add_parser("follow", help="follow one player and label them (name, number)")
    follow.add_argument("target", help="a video file, or a library asset id (or its start)")
    follow.add_argument("--number", required=True, help="the player's jersey number")
    follow.add_argument("--name", help="the player's name, shown in the label")
    follow.add_argument("--at", type=float, required=True,
                        help="a moment (seconds) when the player is clearly visible")
    follow.add_argument("--point", required=True,
                        help="where the player is at that moment: X,Y in video pixels, "
                             "or fractions such as 0.42,0.6")
    _add_detector_args(follow)
    follow.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    follow.add_argument("--fps", type=float, default=15.0,
                        help="frames analysed per second (default 15)")
    _add_window_args(follow)
    follow.add_argument("--height-cm", type=float,
                        help="the player's height, for distances and speeds (default 140)")
    follow.add_argument("--library", type=Path, help=library_help)
    follow.add_argument("--out", type=Path, help="run folder")
    follow.set_defaults(func=cmd_follow)

    analyze = sub.add_parser("analyze", help="re-work a follow run's stats and events")
    analyze.add_argument("target", help="a library asset id (or its start)")
    analyze.add_argument("--run", help="follow run folder name (default: the latest)")
    analyze.add_argument("--goals", help="mark the goals: x1,y1,x2,y2 fractions of the "
                                         "frame, ';' between goals ('' clears them)")
    analyze.add_argument("--height-cm", type=float, help="the player's height")
    analyze.add_argument("--library", type=Path, help=library_help)
    analyze.set_defaults(func=cmd_analyze)

    export = sub.add_parser("export", help="make a player cam or highlight reel video")
    export.add_argument("target", help="a library asset id (or its start)")
    export.add_argument("--run", help="follow run folder name (default: the latest)")
    export.add_argument("--kind", choices=["playercam", "highlights"], default="highlights",
                        help="playercam: the whole game zoomed in on the player; "
                             "highlights: their best moments (default)")
    export.add_argument("--vertical", action="store_true", help="720x1280, for phones")
    export.add_argument("--library", type=Path, help=library_help)
    export.set_defaults(func=cmd_export)

    fix = sub.add_parser("fix", help="follow again with one more tap where it went wrong")
    fix.add_argument("target", help="a library asset id (or its start)")
    fix.add_argument("--run", help="follow run folder name (default: the latest)")
    fix.add_argument("--at", type=float, required=True, help="a moment (seconds) to tap")
    fix.add_argument("--point", required=True,
                     help="where the player really is then: X,Y pixels or fractions")
    fix.add_argument("--no-video", action="store_true", help="skip drawing followed.mp4")
    fix.add_argument("--library", type=Path, help=library_help)
    fix.set_defaults(func=cmd_fix)

    dataset = sub.add_parser(
        "dataset", help="build a training set from labelled frames (training permission only)")
    dataset.add_argument("--name", required=True, help="a new name, e.g. indoor-2026-11")
    dataset.add_argument("--labels", type=Path, action="append",
                         help="a COCO labels file exported from CVAT (repeat for more)")
    dataset.add_argument("--images", type=Path, action="append",
                         help="the folder of the frames those labels are for (same order)")
    dataset.add_argument("--seed", type=int, default=0, help="for choosing test games")
    dataset.add_argument("--library", type=Path, help=library_help)
    dataset.add_argument("--out", type=Path, help="default: data/datasets/<name>")
    dataset.set_defaults(func=cmd_dataset)

    train = sub.add_parser("train-detector",
                           help="fine-tune the detector on a training set, then score it")
    train.add_argument("dataset", type=Path, help="a folder made by `soccer-recorder dataset`")
    train.add_argument("--name", required=True, help="a new name for the model")
    train.add_argument("--size", choices=list(SIZES), default=DEFAULT_SIZE)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=4,
                       help="batches per update: batch size x this is the effective batch")
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--resolution", type=int, help="training input size (default: the "
                                                      "size's own)")
    train.add_argument("--workers", type=int, default=2, help="data loading processes")
    train.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    train.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                       help="frame size when scoring, as for videos")
    train.add_argument("--out", type=Path, help="default: data/models/<name>")
    train.set_defaults(func=cmd_train_detector)

    reviews = sub.add_parser("reviews", help="how often each kind of moment was right, "
                                             "from your ticks; updates the reel's settings")
    reviews.add_argument("--library", type=Path, help=library_help)
    reviews.set_defaults(func=cmd_reviews)

    serve = sub.add_parser("serve", help="open the local website: drop videos, run detection")
    serve.add_argument("--port", type=int, default=8765, help="default 8765")
    serve.add_argument("--library", type=Path, help=library_help)
    serve.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    serve.add_argument("--weights", type=Path,
                       help="detect with a fine-tuned model from train-detector")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (MediaError, IngestError, LabelError, DetectorUnavailable, ValueError,
            LookupError) as error:
        # SeedNotFound is a ValueError: its message tells the user what to do.
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode(errors="replace").strip()
        print(f"error: {' '.join(map(str, error.cmd[:1]))} failed: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
