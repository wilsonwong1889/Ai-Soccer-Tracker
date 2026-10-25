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


@dataclass
class Scene:
    players: list[Player]
    blocker: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 drawn over players
    size: tuple[int, int] = (640, 360)
    fps: float = 15.0
    duration: float = 8.0

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
        """Write the video; returns each player's true box per frame (None when hidden)."""
        truth = {p.name: [] for p in self.players}
        width, height = self.size
        with VideoWriter(path, width, height, self.fps) as writer:
            for t in self.times:
                image, visible = self.render(float(t))
                writer.write(image)
                for player in self.players:
                    truth[player.name].append(
                        player.box(float(t)) if visible[player.name] >= 0.4 else None)
        return truth


class OracleDetector:
    """Returns the true boxes of the players that are mostly visible, frame by frame."""

    name = "oracle"

    def __init__(self, truth: dict[str, list], jitter: float = 1.0, miss_rate: float = 0.0,
                 seed: int = 1, step: int = 1):
        self.frames = [
            [box for boxes in truth.values() if (box := boxes[i]) is not None]
            for i in range(len(next(iter(truth.values()))))
        ][::step]
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
        return [Box(x1 + n[0], y1 + n[1], x2 + n[2], y2 + n[3], 0.9, "person")
                for (x1, y1, x2, y2), n in zip(boxes, noise, strict=True)]

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
