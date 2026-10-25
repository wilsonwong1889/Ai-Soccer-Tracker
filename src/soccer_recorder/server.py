"""A small local website: drop videos in, run detection, watch progress and results.

    uv run soccer-recorder serve        # opens http://127.0.0.1:8765

Standard library only. This is a tool for one person on one computer, not
the product's upload API (that's R8). The footage is of children, so the
server only listens on 127.0.0.1, rejects requests whose Host isn't this
computer (which stops other websites reaching it through DNS tricks), and
requires a custom header on every change, which browsers won't send
cross-site without permission.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .detect import Detector, Progress, run_detection
from .library import Asset, ingest, list_assets, load_asset

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
ACTION_HEADER = "X-Soccer-Recorder"
CHUNK = 1024 * 1024
MAX_JOBS_SHOWN = 50
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".jpg": "image/jpeg",
}
SIZES = ("nano", "small", "medium", "large")


class Cancelled(Exception):
    """The job was cancelled from the page (or the server is stopping)."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    asset_id: str | None = None
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float | None = None
    message: str = ""
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "asset_id": self.asset_id,
            "status": self.status, "progress": self.progress, "message": self.message,
            "error": self.error, "result": self.result, "created": self.created,
            "started": self.started, "finished": self.finished,
        }


class Jobs:
    """Background work. Ingests and detections each run one at a time, in
    separate queues, so a long detection never holds up adding a video."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queues = {
            "ingest": ThreadPoolExecutor(1, thread_name_prefix="ingest"),
            "detect": ThreadPoolExecutor(1, thread_name_prefix="detect"),
        }

    def submit(self, kind: str, title: str, work: Callable[[Job], dict | None],
               asset_id: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title, asset_id=asset_id)
        with self._lock:
            self._jobs[job.id] = job
        self._queues[kind].submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[Job], dict | None]) -> None:
        if job.cancel.is_set():
            job.status, job.finished = "cancelled", time.time()
            return
        job.status, job.started = "running", time.time()
        try:
            job.result = work(job)
            job.status, job.progress = "done", 1.0
        except Cancelled:
            job.status = "cancelled"
        except Exception as error:  # shown on the page; the server keeps going
            job.status, job.error = "failed", str(error) or type(error).__name__
        finally:
            job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.cancel.set()
        return True

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [job.to_dict() for job in jobs[:MAX_JOBS_SHOWN]]

    def wait_idle(self, timeout: float = 30.0) -> None:
        """For tests: wait until nothing is queued or running."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                busy = any(j.status in ("queued", "running") for j in self._jobs.values())
            if not busy:
                return
            time.sleep(0.05)
        raise TimeoutError("jobs still running")

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
        for queue in self._queues.values():
            queue.shutdown(wait=True, cancel_futures=True)


def _safe_name(name: str) -> str:
    name = Path(unquote(name)).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .") or "video"
    return name[-100:]


def _run_summary(asset: Asset, run_dir: Path) -> dict | None:
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    people = metrics.get("detections", {}).get("person", {})
    ball = metrics.get("detections", {}).get("sports ball", {})
    video = run_dir / "annotated.mp4"
    return {
        "name": run_dir.name,
        "created_at": metrics.get("created_at"),
        "detector": metrics.get("detector", {}).get("name"),
        "device": metrics.get("detector", {}).get("device"),
        "seconds": metrics.get("window", {}).get("seconds"),
        "realtime_factor": metrics.get("speed", {}).get("realtime_factor"),
        "players_per_frame": people.get("per_frame"),
        "ball_seen": ball.get("frames_with_any"),
        "player_height_p50": (metrics.get("person_height_px") or {}).get("p50"),
        "video_url": f"/files/{asset.asset_id}/runs/{run_dir.name}/annotated.mp4"
        if video.is_file() else None,
        "metrics_url": f"/files/{asset.asset_id}/runs/{run_dir.name}/metrics.json",
    }


class App:
    """What the website does, separate from HTTP so it can be tested directly."""

    def __init__(self, library: Path, detector_factory: Callable[[str], Detector] | None = None):
        self.library = Path(library)
        self.uploads = self.library.parent / "uploads"
        self.jobs = Jobs()
        self._detector_factory = detector_factory
        self._detectors: dict[str, Detector] = {}

    # --- reading state ---

    def detection_available(self) -> bool:
        if self._detector_factory is not None:
            return True
        return bool(importlib.util.find_spec("rfdetr") and importlib.util.find_spec("torch"))

    def assets(self) -> list[dict]:
        result = []
        for asset in reversed(list_assets(self.library)):
            runs_dir = asset.directory / "runs"
            runs = []
            if runs_dir.is_dir():
                for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                    summary = _run_summary(asset, run_dir) if run_dir.is_dir() else None
                    if summary:
                        runs.append(summary)
            video = asset.video
            result.append({
                "asset_id": asset.asset_id,
                "filename": asset.original_filename,
                "ingested_at": asset.ingested_at,
                "duration_s": video.duration_s,
                "size": [video.display_width, video.display_height],
                "fps": round(video.fps, 2),
                "codec": video.codec,
                "size_bytes": video.size_bytes,
                "device": video.device,
                "warnings": video.warnings,
                "source_url": f"/files/{asset.asset_id}/{asset.source}",
                "runs": runs,
            })
        return result

    def state(self) -> dict:
        return {
            "library": str(self.library.resolve()),
            "detection_available": self.detection_available(),
            "sizes": list(SIZES),
            "assets": self.assets(),
            "jobs": self.jobs.list(),
        }

    # --- actions ---

    def add_upload(self, temp_path: Path, filename: str) -> Job:
        """Queue an ingest of an uploaded file, which then moves into the library."""

        def work(job: Job) -> dict:
            job.message = "Checking and hashing the video"
            try:
                asset, created = ingest(temp_path, self.library, move=True,
                                        original_filename=filename)
            finally:
                temp_path.unlink(missing_ok=True)  # duplicate or failure: drop the upload
            job.asset_id = asset.asset_id
            job.message = "Added to the library" if created else "Already in the library"
            return {"asset_id": asset.asset_id, "created": created}

        return self.jobs.submit("ingest", f"Add {filename}", work)

    def _detector(self, size: str) -> Detector:
        if size not in self._detectors:
            if self._detector_factory is not None:
                self._detectors[size] = self._detector_factory(size)
            else:
                from .rfdetr_detector import RFDETRDetector

                self._detectors[size] = RFDETRDetector(size)
        return self._detectors[size]

    def start_detection(self, asset_id: str, size: str = "small",
                        seconds: float | None = None) -> Job:
        if size not in SIZES:
            raise ValueError(f"size must be one of {', '.join(SIZES)}")
        if seconds is not None and seconds <= 0:
            raise ValueError("seconds must be positive")
        directory = self.library / asset_id
        known = re.fullmatch(r"[0-9a-f]{16}", asset_id) and (directory / "manifest.json").is_file()
        if not known:
            raise LookupError(f"no video {asset_id!r} in the library")
        asset = load_asset(directory)
        label = "whole video" if seconds is None else f"first {seconds / 60:g} min"

        def work(job: Job) -> dict:
            job.message = f"Loading RF-DETR {size}"
            detector = self._detector(size)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = asset.directory / "runs" / f"detect-{detector.name}-{stamp}"

            def on_progress(progress: Progress) -> None:
                if job.cancel.is_set():
                    raise Cancelled()
                total = progress.total_video_s or 1.0
                job.progress = min(1.0, progress.video_s / total)
                rate = progress.video_s / progress.elapsed_s if progress.elapsed_s else 0
                eta = (total - progress.video_s) / rate if rate else None
                job.message = (
                    f"{progress.video_s:.0f} / {total:.0f} s of video"
                    + (f" · about {eta / 60:.1f} min left" if eta is not None else "")
                )

            try:
                metrics = run_detection(asset.source_path, detector, out,
                                        duration_s=seconds, on_progress=on_progress)
            except BaseException:
                shutil.rmtree(out, ignore_errors=True)  # don't leave half a run behind
                raise
            speed = metrics["speed"]["realtime_factor"]
            job.message = f"Done in {metrics['speed']['wall_s']:.0f} s ({speed}× real time)"
            return {"run": out.name}

        return self.jobs.submit("detect", f"Detect players in {asset.original_filename} "
                                f"({label}, {size})", work, asset_id=asset_id)

    def file_path(self, asset_id: str, relative: str) -> Path | None:
        """A file inside an asset folder, or None if it isn't allowed."""
        if not re.fullmatch(r"[0-9a-f]{16}", asset_id):
            return None
        parts = Path(relative).parts
        if not parts or any(part in ("..", "") or part.startswith(".") for part in parts):
            return None
        directory = (self.library / asset_id).resolve()
        candidate = self.library / asset_id / relative
        if Path(relative).suffix.lower() not in CONTENT_TYPES or not candidate.is_file():
            return None
        if len(parts) == 1 and parts[0].startswith("source."):
            return candidate.resolve()  # the original may be a symlink (ingest --link)
        resolved = candidate.resolve()
        return resolved if resolved.is_relative_to(directory) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "SoccerRecorder"
    app: App  # set on the subclass made by make_server

    def log_message(self, format, *args):  # noqa: A002 - keep the terminal quiet
        pass

    # --- helpers ---

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ALLOWED_HOSTS

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _send_file(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match or not (match[1] or match[2]):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix.lower()])
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = file.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browsers drop video range requests all the time

    # --- routes ---

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        if not self._host_ok():
            return self._error(403, "This server only answers on this computer.")
        url = urlparse(self.path)
        if url.path == "/":
            page = importlib.resources.files("soccer_recorder").joinpath("static/index.html")
            body = page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return None
        if url.path == "/api/state":
            return self._send_json(self.app.state())
        match = re.fullmatch(r"/files/([0-9a-f]{16})/(.+)", url.path)
        if match:
            path = self.app.file_path(match[1], unquote(match[2]))
            if path is None:
                return self._error(404, "Not found")
            return self._send_file(path)
        return self._error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "This server only answers on this computer.")
        if self.headers.get(ACTION_HEADER) != "1":
            return self._error(403, f"Missing {ACTION_HEADER} header.")
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                return self._upload(url)
            if url.path == "/api/detect":
                body = self._json_body()
                seconds = body.get("seconds")
                job = self.app.start_detection(
                    str(body.get("asset_id", "")), str(body.get("size", "small")),
                    float(seconds) if seconds else None,
                )
                return self._send_json({"job": job.id}, 202)
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", url.path)
            if match:
                if not self.app.jobs.cancel(match[1]):
                    return self._error(409, "That job isn't queued or running.")
                return self._send_json({"ok": True})
        except LookupError as error:
            return self._error(404, str(error))
        except ValueError as error:
            return self._error(400, str(error))
        return self._error(404, "Not found")

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64 * 1024:
            raise ValueError("request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ValueError("body must be JSON") from None
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _upload(self, url) -> None:
        """The file is the raw request body (no form encoding), streamed to disk."""
        name = _safe_name(parse_qs(url.query).get("name", ["video"])[0])
        try:
            length = int(self.headers["Content-Length"])
        except (TypeError, ValueError):
            return self._error(411, "Content-Length is required")
        if length <= 0:
            return self._error(400, "The file is empty")
        uploads = self.app.uploads
        uploads.mkdir(parents=True, exist_ok=True)
        temp = uploads / f".{uuid.uuid4().hex}-{name}.part"
        try:
            with temp.open("wb") as file:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        raise ConnectionError("upload interrupted")
                    file.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionError, OSError):
            temp.unlink(missing_ok=True)
            return None  # the browser has gone; nobody to answer
        final = uploads / f"{uuid.uuid4().hex[:8]}-{name}"
        temp.rename(final)
        job = self.app.add_upload(final, name)
        return self._send_json({"job": job.id}, 202)


def make_server(app: App, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server
