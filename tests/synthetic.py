"""Synthetic game scenes with known ground truth, for testing tracking and following.

Players are drawn as head (hair), shirt, shorts and socks on a green pitch,
moving along known paths. A grey block hides whoever walks behind it. The
"oracle" detector returns the true boxes of players that are mostly visible,
which is what a good detector would see.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from soccer_recorder.detect import Box
from soccer_recorder.render import VideoWriter

PITCH = (50, 125, 46)  # BGR
RED, BLUE = (40, 40, 220), (200, 80, 30)
WHITE, BLACK = (235, 235, 235), (25, 25, 25)
BLOND, DARK_HAIR, SKIN = (90, 200, 230), (30, 35, 45), (140, 170, 210)
GREY = (150, 150, 150)


@dataclass
class Player:
    name: str
    shirt: tuple
    shorts: tuple
    socks: tuple
    hair: tuple
    path: Callable[[float], tuple[float, float]]  # t -> (centre x, feet y)
    width: int = 22
    height: int = 60

    def box(self, t: float) -> tuple[int, int, int, int]:
        cx, feet = self.path(t)
        x1 = int(round(cx - self.width / 2))
        y1 = int(round(feet - self.height))
        return x1, y1, x1 + self.width, int(round(feet))


def draw_player(image: np.ndarray, player: Player, t: float, mask: np.ndarray, value: int):
    x1, y1, x2, y2 = player.box(t)
    h = y2 - y1
    bands = [
        (0.00, 0.10, player.hair, 0.6),
        (0.10, 0.18, SKIN, 0.5),
        (0.18, 0.55, player.shirt, 1.0),
        (0.55, 0.72, player.shorts, 0.9),
        (0.72, 0.84, SKIN, 0.5),
        (0.84, 1.00, player.socks, 0.6),
    ]
    height, width = image.shape[:2]
    for top, bottom, colour, part_width in bands:
        inset = int(round((1 - part_width) * (x2 - x1) / 2))
        ya, yb = max(0, y1 + int(top * h)), min(height, y1 + int(bottom * h))
        xa, xb = max(0, x1 + inset), min(width, x2 - inset)
        if ya < yb and xa < xb:
            image[ya:yb, xa:xb] = colour
            mask[ya:yb, xa:xb] = value


BALL_KEY = "__ball__"


@dataclass
class Scene:
    players: list[Player]
    blocker: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 drawn over players
    size: tuple[int, int] = (640, 360)
    fps: float = 15.0
    duration: float = 8.0
    ball: Callable[[float], tuple[float, float] | None] | None = None  # t -> centre
    goals: tuple[tuple[int, int, int, int], ...] = ()  # drawn as white frames

    @property
    def times(self) -> np.ndarray:
        return np.arange(int(round(self.duration * self.fps))) / self.fps

    def render(self, t: float) -> tuple[np.ndarray, dict[str, float]]:
        """The frame at t, and how much of each player is visible (0 to 1)."""
        width, height = self.size
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = PITCH
        image[height // 2 - 1:height // 2 + 1, :] = WHITE  # a pitch line
        mask = np.zeros((height, width), dtype=np.int16)
        # Nearer players (lower on screen) are drawn last, over farther ones.
        order = sorted(range(len(self.players)), key=lambda i: self.players[i].path(t)[1])
        for i in order:
            draw_player(image, self.players[i], t, mask, i + 1)
        for x1, y1, x2, y2 in self.goals:
            image[y1:y1 + 2, x1:x2] = WHITE
            image[y2 - 2:y2, x1:x2] = WHITE
            image[y1:y2, x1:x1 + 2] = WHITE
            image[y1:y2, x2 - 2:x2] = WHITE
        if self.ball and (centre := self.ball(t)) is not None:
            bx, by = (int(round(v)) for v in centre)
            image[max(0, by - 4):by + 4, max(0, bx - 4):bx + 4] = WHITE
            image[max(0, by - 1):by + 1, max(0, bx - 4):bx + 4] = BLACK  # a seam
        if self.blocker:
            x1, y1, x2, y2 = self.blocker
            image[y1:y2, x1:x2] = GREY
            mask[y1:y2, x1:x2] = 0
        visible = {}
        for i, player in enumerate(self.players):
            x1, y1, x2, y2 = player.box(t)
            full = np.zeros_like(mask)
            draw_player(np.zeros_like(image), player, t, full, 1)
            total = max(int(full.sum()), 1)
            visible[player.name] = float(((mask == i + 1) & (full == 1)).sum()) / total
        return image, visible

    def write(self, path) -> dict[str, list]:
        """Write the video; returns each player's true box per frame (None when hidden).

        With a ball, truth[BALL_KEY] holds its box per frame (None when off the pitch).
        """
        truth = {p.name: [] for p in self.players}
        if self.ball:
            truth[BALL_KEY] = []
        width, height = self.size
        with VideoWriter(path, width, height, self.fps) as writer:
            for t in self.times:
                image, visible = self.render(float(t))
                writer.write(image)
                for player in self.players:
                    truth[player.name].append(
                        player.box(float(t)) if visible[player.name] >= 0.4 else None)
                if self.ball:
                    centre = self.ball(float(t))
                    truth[BALL_KEY].append(None if centre is None else (
                        centre[0] - 4, centre[1] - 4, centre[0] + 4, centre[1] + 4))
        return truth


class OracleDetector:
    """Returns the true boxes of the players that are mostly visible, frame by frame."""

    name = "oracle"

    def __init__(self, truth: dict[str, list], jitter: float = 1.0, miss_rate: float = 0.0,
                 seed: int = 1, step: int = 1):
        people = {k: v for k, v in truth.items() if k != BALL_KEY}
        self.frames = [
            [box for boxes in people.values() if (box := boxes[i]) is not None]
            for i in range(len(next(iter(people.values()))))
        ][::step]
        self.balls = truth.get(BALL_KEY, [])[::step]
        self.calls = 0
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter
        self.miss_rate = miss_rate

    def detect(self, image):
        boxes = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        if self.miss_rate:
            boxes = [b for b in boxes if self.rng.random() >= self.miss_rate]
        noise = self.rng.normal(0, self.jitter, (len(boxes), 4))
        found = [Box(x1 + n[0], y1 + n[1], x2 + n[2], y2 + n[3], 0.9, "person")
                 for (x1, y1, x2, y2), n in zip(boxes, noise, strict=True)]
        index = self.calls - 1
        if self.balls and index < len(self.balls) and (ball := self.balls[index]) is not None:
            if not self.miss_rate or self.rng.random() >= self.miss_rate:
                found.append(Box(*ball, 0.7, "sports ball"))
        return found

    def describe(self):
        return {"name": self.name, "device": "cpu"}


def blocker_scene() -> Scene:
    """The target walks behind a block for ~2 s while a teammate crosses the other way."""
    return Scene(
        players=[
            # The child we follow: red shirt, white shorts, red socks, dark hair.
            Player("target", RED, WHITE, RED, DARK_HAIR, lambda t: (100 + 60 * t, 250)),
            # Teammate in the same kit; only the hair differs. Crosses behind the block.
            Player("teammate", RED, WHITE, RED, BLOND, lambda t: (540 - 60 * t, 252)),
            # A look-alike teammate (same colours as the target) far away.
            Player("lookalike", RED, WHITE, RED, DARK_HAIR, lambda t: (120 + 30 * t, 120)),
            # An opponent near where the target comes out.
            Player("opponent", BLUE, BLUE, BLUE, DARK_HAIR, lambda t: (470 - 10 * t, 300)),
        ],
        blocker=(260, 150, 400, 280),
    )


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crossing_scene() -> Scene:
    """Players cross the target directly: an opponent in front of them, then a teammate behind."""
    return Scene(
        players=[
            Player("target", RED, WHITE, RED, DARK_HAIR, lambda t: (100 + 70 * t, 250)),
            # Crosses in front of the target around t = 3.1 s.
            Player("opponent", BLUE, BLUE, BLUE, DARK_HAIR, lambda t: (540 - 70 * t, 254)),
            # Same kit as the target (only the hair differs); crosses behind them at ~5.5 s.
            Player("teammate", RED, WHITE, RED, BLOND, lambda t: (870 - 70 * t, 248)),
        ],
    )


def _lerp(t: float, t0: float, t1: float, a: tuple, b: tuple) -> tuple[float, float]:
    w = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def match_scene() -> Scene:
    """A scripted attack with a known answer.

    0.0-2.0 s  the target dribbles right (130 px, over 2 body heights)
    2.0-2.6 s  passes to the teammate
    2.6-4.0 s  the teammate has the ball
    4.0-4.6 s  passes back to the target
    4.6-6.5 s  the target dribbles towards the right goal
    6.5-6.9 s  shoots: the ball flies into the right goal and stays there
    7.2 s      the crowd cheers (see match_audio); 8.5 s: the whistle
    """
    def target_path(t):
        if t < 2.0:
            return (100 + 65 * t, 260)
        if t < 4.6:
            return (230 + 30 * (t - 2.0) / 2.6, 260)  # drifts to x = 260
        if t < 6.5:
            return (260 + 100 * (t - 4.6), 260)  # to x = 450
        return (450 + 20 * (t - 6.5), 260)

    teammate = lambda t: (330 + 5 * t, 232)  # noqa: E731

    def ball(t):
        tx, ty = target_path(t)
        mx, my = teammate(t)
        if t < 2.0:
            return (tx + 12, ty - 6)
        if t < 2.6:
            return _lerp(t, 2.0, 2.6, (target_path(2.0)[0] + 12, 254), (teammate(2.6)[0] + 10, 226))
        if t < 4.0:
            return (mx + 10, my - 6)
        if t < 4.6:
            return _lerp(t, 4.0, 4.6, (teammate(4.0)[0] + 10, 226), (target_path(4.6)[0] + 12, 254))
        if t < 6.5:
            return (tx + 12, ty - 6)
        if t < 6.9:
            return _lerp(t, 6.5, 6.9, (target_path(6.5)[0] + 12, 254), (622, 205))
        return (622 + 2 * np.sin(t * 3), 205)  # resting in the net

    return Scene(
        players=[
            Player("target", RED, WHITE, RED, DARK_HAIR, target_path),
            Player("teammate", RED, WHITE, RED, BLOND, teammate),
            Player("opponent", BLUE, BLACK, BLUE, DARK_HAIR, lambda t: (520 - 10 * t, 300)),
            Player("keeper", BLUE, BLACK, BLUE, BLOND, lambda t: (585, 240 + 10 * np.sin(t))),
        ],
        duration=10.0,
        ball=ball,
        goals=((0, 150, 30, 262), (608, 150, 640, 262)),
    )


def match_audio(path, duration: float, cheer=(7.2, 10.0), whistle=(8.5, 9.1),
                rate: int = 16000):
    """Write a WAV: crowd murmur, a cheer and a referee's whistle."""
    import wave

    rng = np.random.default_rng(0)
    t = np.arange(int(rate * duration)) / rate
    audio = rng.normal(0, 0.01, len(t))
    loud = (t >= cheer[0]) & (t < cheer[1])
    audio[loud] += rng.normal(0, 0.12, loud.sum())
    blow = (t >= whistle[0]) & (t < whistle[1])
    audio[blow] += 0.2 * np.sin(2 * np.pi * 3200 * t[blow])
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def with_audio(video, wav, out):
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(out)], check=True)
    return out
