"""A simple appearance description of a player: colour, by body part.

Each player box is cut into three bands: head/hair, shirt, and shorts/socks.
Each band gets a small colour histogram (hue and saturation for colourful
pixels, light/dark for black, grey and white ones). Teammates share a shirt,
so the other two bands are what tells them apart. It's crude next to a
trained re-identification model (R5), but it's free, fast, and has no
model weights or training data with licensing questions.
"""

from __future__ import annotations

import numpy as np

# Vertical bands as fractions of the box height, and how much each counts.
PARTS = (
    ("head", 0.00, 0.18, 0.2),
    ("shirt", 0.18, 0.55, 0.5),
    ("legs", 0.55, 0.90, 0.3),
)
HUE_BINS = 8
SAT_BINS = 3
GREY_BINS = 3  # dark, mid, light, for pixels with little colour
PART_SIZE = HUE_BINS * SAT_BINS + GREY_BINS
FEATURE_SIZE = PART_SIZE * len(PARTS)
MIN_PART_PIXELS = 4


def bgr_to_hsv(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in [0, 1), saturation and value in [0, 1], for BGR uint8 pixels."""
    rgb = pixels[..., ::-1].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    chroma = value - low
    saturation = np.where(value > 0, chroma / np.maximum(value, 1e-6), 0.0)
    safe = np.maximum(chroma, 1e-6)
    hue = np.where(
        value == r, ((g - b) / safe) % 6,
        np.where(value == g, (b - r) / safe + 2, (r - g) / safe + 4),
    ) / 6.0
    hue = np.where(chroma > 0, hue % 1.0, 0.0)
    return hue, saturation, value


def _part_histogram(pixels: np.ndarray) -> np.ndarray | None:
    if pixels.size == 0 or pixels.shape[0] * pixels.shape[1] < MIN_PART_PIXELS:
        return None
    hue, sat, val = (channel.ravel() for channel in bgr_to_hsv(pixels))
    colourful = (sat >= 0.25) & (val >= 0.2)
    hist = np.zeros(PART_SIZE, dtype=np.float32)
    hue_bin = np.minimum((hue[colourful] * HUE_BINS).astype(int), HUE_BINS - 1)
    sat_bin = np.minimum(((sat[colourful] - 0.25) / 0.75 * SAT_BINS).astype(int), SAT_BINS - 1)
    np.add.at(hist, hue_bin * SAT_BINS + sat_bin, 1)
    grey_bin = np.minimum((val[~colourful] * GREY_BINS).astype(int), GREY_BINS - 1)
    np.add.at(hist, HUE_BINS * SAT_BINS + grey_bin, 1)
    return hist / hist.sum()


def describe(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Feature vector for the player in `box` (image pixels), or None if too small."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 3 or box_h < 6:
        return None
    # The middle 60% of the width: mostly player, not the floor beside them.
    left = int(max(0, x1 + 0.2 * box_w))
    right = int(min(width, x2 - 0.2 * box_w))
    parts = []
    for _, top, bottom, _ in PARTS:
        y_top = int(max(0, y1 + top * box_h))
        y_bottom = int(min(height, y1 + bottom * box_h))
        hist = _part_histogram(image[y_top:y_bottom, left:right])
        if hist is None:
            return None
        parts.append(hist)
    return np.concatenate(parts)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """0 (nothing alike) to 1 (same colours), weighting the parts as in PARTS."""
    if a is None or b is None:
        return 0.0
    total = 0.0
    for i, (_, _, _, weight) in enumerate(PARTS):
        part_a = a[i * PART_SIZE:(i + 1) * PART_SIZE]
        part_b = b[i * PART_SIZE:(i + 1) * PART_SIZE]
        total += weight * float(np.sum(np.sqrt(part_a * part_b)))  # Bhattacharyya
    return total


def mean_feature(features: list[np.ndarray]) -> np.ndarray | None:
    usable = [f for f in features if f is not None]
    if not usable:
        return None
    return np.mean(np.stack(usable), axis=0)
