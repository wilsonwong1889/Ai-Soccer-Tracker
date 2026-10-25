"""Split the tracked players into two teams by shirt colour.

Each tracklet's average shirt histogram is clustered into two groups
(k-means). Players far from both groups (a referee, a keeper in a different
kit) are "other". The followed player's team is whichever group their own
shirt is closest to. It's the same idea as Roboflow's team classifier, with
plain colour histograms instead of a neural network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import PART_SIZE

SHIRT = slice(PART_SIZE, 2 * PART_SIZE)  # the shirt band of an appearance feature
MIN_FRAMES = 5
OTHER = -1


@dataclass
class Teams:
    centres: np.ndarray  # (2, PART_SIZE)
    of_track: dict[int, int]  # track id -> 0, 1 or OTHER

    def team_of(self, shirt: np.ndarray | None) -> int:
        if shirt is None or len(self.centres) < 2:
            return OTHER
        distances = np.linalg.norm(self.centres - shirt, axis=1)
        return int(np.argmin(distances))


def _kmeans(points: np.ndarray, k: int = 2, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means: start from the two points furthest apart."""
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
    centres = points[[first, second]].astype(float)
    labels = np.zeros(len(points), dtype=int)
    for step in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if step > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            if np.any(labels == c):
                centres[c] = points[labels == c].mean(axis=0)
    return centres, labels


def split_teams(track_ids: np.ndarray, features: np.ndarray, other_factor: float = 2.5) -> Teams:
    """Teams from per-box track ids and appearance features (as in Tracks)."""
    shirts, ids = [], []
    for track_id in np.unique(track_ids):
        rows = features[track_ids == track_id][:, SHIRT]
        rows = rows[~np.isnan(rows[:, 0])]
        if len(rows) >= MIN_FRAMES:
            shirts.append(rows.mean(axis=0))
            ids.append(int(track_id))
    if len(shirts) < 2:
        return Teams(centres=np.zeros((0, PART_SIZE)), of_track={i: OTHER for i in ids})
    points = np.stack(shirts)
    centres, labels = _kmeans(points)
    distances = np.linalg.norm(points - centres[labels], axis=1)
    typical = float(np.median(distances)) or 1e-6
    of_track = {}
    for track_id, label, distance in zip(ids, labels, distances, strict=True):
        of_track[track_id] = int(label) if distance <= other_factor * typical + 0.05 else OTHER
    return Teams(centres=centres, of_track=of_track)
