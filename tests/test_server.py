import http.client
import json
import threading
import time

import pytest
from conftest import FixedDetector, needs_ffmpeg

from soccer_recorder.server import App, make_server

pytestmark = needs_ffmpeg


class SlowDetector(FixedDetector):
    def detect(self, image):
        time.sleep(0.05)
        return super().detect(image)


@pytest.fixture
def site(tmp_path):
    detectors = {"fixed": FixedDetector, "slow": SlowDetector}
    choice = {"kind": "fixed"}
    app = App(tmp_path / "library", detector_factory=lambda size: (
        choice["make"]() if "make" in choice else detectors[choice["kind"]]()))
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None, action=True):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = dict(headers or {})
        if action:
            headers["X-Soccer-Recorder"] = "1"
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def state():
        return json.loads(request("GET", "/api/state")[1])

    yield app, request, state, choice
    server.shutdown()
    server.server_close()
    app.jobs.shutdown()


def upload(request, path, name):
    response, body = request("POST", f"/api/upload?name={name}", body=path.read_bytes())
    assert response.status == 202, body
    return json.loads(body)["job"]


def test_page_and_empty_state(site):
    app, request, state, _ = site
    response, body = request("GET", "/")
    assert response.status == 200
    assert b"Drop game videos here" in body
    current = state()
    assert current["assets"] == [] and current["jobs"] == []
    assert current["detection_available"] is True


def test_other_hosts_and_missing_header_are_refused(site, landscape_clip):
    _, request, _, _ = site
    response, _ = request("GET", "/api/state", headers={"Host": "evil.example"})
    assert response.status == 403
    response, _ = request("POST", "/api/upload?name=x.mp4", body=b"x", action=False)
    assert response.status == 403


def test_upload_adds_the_video_and_cleans_up(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "My%20Game.mp4")
    app.jobs.wait_idle()
    current = state()
    assert [a["filename"] for a in current["assets"]] == ["My Game.mp4"]
    job = current["jobs"][0]
    assert job["status"] == "done" and job["message"] == "Added to the library"
    assert list(app.uploads.iterdir()) == []  # moved into the library, nothing left over
    asset = current["assets"][0]
    assert asset["size"] == [320, 180] and asset["runs"] == []

    upload(request, landscape_clip, "copy.mp4")  # same bytes again
    app.jobs.wait_idle()
    current = state()
    assert len(current["assets"]) == 1
    assert current["jobs"][0]["message"] == "Already in the library"
    assert list(app.uploads.iterdir()) == []


def test_upload_of_a_non_video_fails_cleanly(site, tmp_path):
    app, request, state, _ = site
    notes = tmp_path / "notes.mp4"
    notes.write_text("not a video")
    upload(request, notes, "notes.mp4")
    app.jobs.wait_idle()
    current = state()
    assert current["assets"] == []
    assert current["jobs"][0]["status"] == "failed"
    assert current["jobs"][0]["error"]
    assert list(app.uploads.iterdir()) == []


def test_detect_then_watch_with_seeking(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]

    response, body = request("POST", "/api/detect",
                             {"asset_id": asset_id, "size": "nano", "seconds": 1})
    assert response.status == 202, body
    app.jobs.wait_idle()
    current = state()
    job = current["jobs"][0]
    assert job["status"] == "done", job
    assert job["progress"] == 1.0 and "real time" in job["message"]
    run = current["assets"][0]["runs"][0]
    assert run["players_per_frame"] == 2.0
    assert run["seconds"] == pytest.approx(1.0, abs=0.1)

    response, video = request("GET", run["video_url"])
    assert response.status == 200
    assert response.getheader("Content-Type") == "video/mp4"
    assert response.getheader("Accept-Ranges") == "bytes"
    size = len(video)

    response, part = request("GET", run["video_url"], headers={"Range": "bytes=0-99"})
    assert response.status == 206 and part == video[:100]
    assert response.getheader("Content-Range") == f"bytes 0-99/{size}"
    response, part = request("GET", run["video_url"], headers={"Range": "bytes=-10"})
    assert response.status == 206 and part == video[-10:]
    response, _ = request("GET", run["video_url"], headers={"Range": f"bytes={size}-"})
    assert response.status == 416

    response, metrics = request("GET", run["metrics_url"])
    assert response.status == 200 and json.loads(metrics)["frames"] > 0
    response, _ = request("GET", current["assets"][0]["source_url"])
    assert response.status == 200


def test_detect_rejects_bad_requests(site):
    _, request, _, _ = site
    response, _ = request("POST", "/api/detect", {"asset_id": "0123456789abcdef"})
    assert response.status == 404
    response, _ = request("POST", "/api/detect", {"asset_id": "x", "size": "xlarge"})
    assert response.status == 400
    response, _ = request("POST", "/api/detect", body=b"not json",
                          headers={"Content-Length": "8"})
    assert response.status == 400


def test_cancel_a_running_detection(site, landscape_clip):
    app, request, state, choice = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    choice["kind"] = "slow"

    response, body = request("POST", "/api/detect", {"asset_id": asset_id, "size": "small"})
    job_id = json.loads(body)["job"]
    for _ in range(100):
        if app.jobs.get(job_id).status == "running" and app.jobs.get(job_id).progress:
            break
        time.sleep(0.02)
    response, _ = request("POST", f"/api/jobs/{job_id}/cancel")
    assert response.status == 200
    app.jobs.wait_idle()
    assert app.jobs.get(job_id).status == "cancelled"
    assert state()["assets"][0]["runs"] == []
    runs = app.library / asset_id / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []  # the half-done run is removed

    response, _ = request("POST", f"/api/jobs/{job_id}/cancel")
    assert response.status == 409  # already finished


@pytest.mark.parametrize("path", [
    "/files/0123456789abcdef/../manifest.json",
    "/files/0123456789abcdef/runs/../../secret.json",
    "/files/not-an-id/manifest.json",
    "/files/0123456789abcdef/.hidden.json",
])
def test_files_outside_assets_are_not_served(site, path):
    _, request, _, _ = site
    response, _ = request("GET", path)
    assert response.status == 404


def test_only_known_file_types_are_served(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    (app.library / asset_id / "notes.txt").write_text("private")
    response, _ = request("GET", f"/files/{asset_id}/notes.txt")
    assert response.status == 404
    response, _ = request("GET", f"/files/{asset_id}/manifest.json")
    assert response.status == 200


has_follow_deps = all(
    __import__("importlib").util.find_spec(name) for name in ("trackers", "PIL")
)


def test_frame_endpoint_returns_a_still(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    response, body = request("GET", f"/api/frame?asset={asset_id}&t=1.2")
    assert response.status == 200
    assert response.getheader("Content-Type") == "image/jpeg"
    assert body[:2] == b"\xff\xd8"  # JPEG
    response, _ = request("GET", "/api/frame?asset=0123456789abcdef&t=1")
    assert response.status == 404


@pytest.mark.skipif(not has_follow_deps, reason="detect extra not installed")
def test_follow_a_player_end_to_end(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    # FixedDetector's first player is at (10, 10)-(30, 70) on the 320x180 frame.
    tap = {"asset_id": asset_id, "number": "10", "name": "Emma", "t": 1.0,
           "x": 20 / 320, "y": 40 / 180, "seconds": 2, "size": "small"}
    response, body = request("POST", "/api/follow", tap)
    assert response.status == 202, body
    app.jobs.wait_idle()
    current = state()
    job = current["jobs"][0]
    assert job["status"] == "done", job
    assert job["message"].startswith("Found Emma  #10")
    run = current["assets"][0]["runs"][0]
    assert run["kind"] == "follow" and run["label"] == "Emma #10"
    assert run["seen"] > 0.8
    response, video = request("GET", run["video_url"])
    assert response.status == 200 and len(video) > 1000
    # Cached tracking isn't listed as a result.
    assert all(r["kind"] != "tracks" for r in current["assets"][0]["runs"])

    response, _ = request("POST", "/api/follow", {**tap, "number": "7", "name": ""})
    assert response.status == 202
    app.jobs.wait_idle()
    job = state()["jobs"][0]
    assert job["status"] == "done" and "reused earlier tracking" in job["message"]
    assert state()["assets"][0]["runs"][0]["label"] == "#7"


@pytest.mark.skipif(not has_follow_deps, reason="detect extra not installed")
def test_follow_tap_on_nobody_fails_with_advice(site, landscape_clip):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    request("POST", "/api/follow", {"asset_id": asset_id, "number": "3", "t": 1.0,
                                    "x": 0.9, "y": 0.9, "seconds": 2})
    app.jobs.wait_idle()
    job = state()["jobs"][0]
    assert job["status"] == "failed"
    assert "No player was found where you tapped" in job["error"]
    assert not any(r["kind"] == "follow" for r in state()["assets"][0]["runs"])


@pytest.mark.parametrize("change, status", [
    ({"number": "ten"}, 400),
    ({"number": "1234"}, 400),
    ({"x": 1.5}, 400),
    ({"t": 90, "seconds": 60}, 400),
    ({"size": "xlarge"}, 400),
    ({"asset_id": "0123456789abcdef"}, 404),
])
def test_follow_rejects_bad_requests(site, landscape_clip, change, status):
    app, request, state, _ = site
    upload(request, landscape_clip, "game.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    body = {"asset_id": asset_id, "number": "10", "t": 1.0, "x": 0.5, "y": 0.5, **change}
    response, _ = request("POST", "/api/follow", body)
    assert response.status == status



@pytest.mark.skipif(not has_follow_deps, reason="detect extra not installed")
def test_analytics_reviews_and_marking_goals(site, tmp_path):
    from synthetic import OracleDetector, match_audio, match_scene, with_audio

    app, request, state, choice = site
    scene = match_scene()
    truth = scene.write(tmp_path / "silent.mp4")
    match_audio(tmp_path / "sound.wav", scene.duration)
    video = with_audio(tmp_path / "silent.mp4", tmp_path / "sound.wav", tmp_path / "match.mp4")
    choice["make"] = lambda: OracleDetector(truth)
    upload(request, video, "match.mp4")
    app.jobs.wait_idle()
    asset_id = state()["assets"][0]["asset_id"]
    x1, y1, x2, y2 = scene.players[0].box(0.5)
    width, height = scene.size
    response, body = request("POST", "/api/follow", {
        "asset_id": asset_id, "number": "10", "name": "Emma", "t": 0.5,
        "x": (x1 + x2) / 2 / width, "y": (y1 + y2) / 2 / height, "height_cm": 140})
    assert response.status == 202, body
    app.jobs.wait_idle()
    run = state()["assets"][0]["runs"][0]
    assert run["kind"] == "follow" and run["stats"]["passes_completed"] == 1
    assert run["stats"]["goals"] == 0  # goals not marked yet

    response, body = request("GET", f"/api/analytics?asset={asset_id}&run={run['name']}")
    assert response.status == 200
    data = json.loads(body)
    kinds = [e["type"] for e in data["events"]]
    assert "possible_goal" in kinds and "goal" not in kinds
    assert data["video_url"].endswith("followed.mp4") and data["settings"]["player_height_m"] == 1.4

    dribble = next(e for e in data["events"] if e["type"] == "dribble")
    response, body = request("POST", "/api/review", {"asset_id": asset_id, "run": run["name"],
                                                     "event": dribble["id"], "verdict": "yes"})
    assert response.status == 200 and json.loads(body)["reviews"] == {dribble["id"]: "yes"}

    goals = [[g[0] / width, g[1] / height, g[2] / width, g[3] / height] for g in scene.goals]
    response, body = request("POST", "/api/goals", {"asset_id": asset_id, "goals": goals})
    assert response.status == 202, body
    app.jobs.wait_idle()
    assert state()["jobs"][0]["status"] == "done"
    data = json.loads(request("GET", f"/api/analytics?asset={asset_id}&run={run['name']}")[1])
    goal = next(e for e in data["events"] if e["type"] == "goal")
    assert goal["confidence"] >= 0.9 and goal["details"]["by_followed_player"]
    assert data["reviews"] == {dribble["id"]: "yes"}  # reviews survive re-analysis
    assert data["review_summary"]["dribble"]["confirmed"] == 1
    assert state()["assets"][0]["runs"][0]["stats"]["goals"] == 1

    response, _ = request("POST", "/api/goals", {"asset_id": asset_id, "goals": [[0, 0, 2, 2]]})
    assert response.status == 400
    response, _ = request("GET", f"/api/analytics?asset={asset_id}&run=follow-n1-20000101-000000")
    assert response.status == 404
    response, _ = request("POST", "/api/review", {"asset_id": asset_id, "run": run["name"],
                                                  "event": "nope-1", "verdict": "yes"})
    assert response.status == 404
