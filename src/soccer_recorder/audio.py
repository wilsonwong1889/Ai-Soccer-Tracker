"""Listen to the game: cheering and the referee's whistle.

No model, just signal processing on the video's sound, read through FFmpeg
in chunks so a whole game never sits in memory:

- **Cheering**: the sound is much louder than the game's running background
  level (a rolling median) for at least a second. Goals usually bring the
  loudest of these, but so do near misses and saves, so a cheer is a hint,
  not a goal.
- **Whistle**: a strong, pure tone between 2 and 5 kHz (where referees'
  whistles sit) for at least 0.15 s.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .media import probe, require_tool

RATE = 16000
HOP = 512  # 32 ms
WINDOW = 1024  # 64 ms
CHUNK_S = 30.0
WHISTLE_BAND = (2000.0, 5000.0)


@dataclass
class Sound:
    start: float
    end: float
    strength: float  # cheers: dB above the background; whistles: tone share of the band
    frequency: float | None = None

    def to_dict(self) -> dict:
        result = {"start": round(float(self.start), 2), "end": round(float(self.end), 2),
                  "strength": round(float(self.strength), 2)}
        if self.frequency is not None:
            result["frequency_hz"] = int(round(self.frequency))
        return result


@dataclass
class Listening:
    has_audio: bool
    cheers: list[Sound] = field(default_factory=list)
    whistles: list[Sound] = field(default_factory=list)
    # Loudness above the background, twice a second, for the timeline.
    loudness: list[float] = field(default_factory=list)
    loudness_step_s: float = 0.5


def read_audio(path: str | Path, start_s: float = 0.0, duration_s: float | None = None):
    """Yield chunks of mono float32 samples at RATE."""
    command = [require_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += ["-i", str(path)]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.3f}"]
    command += ["-map", "0:a:0", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunk_bytes = int(CHUNK_S * RATE) * 2
    try:
        while True:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            yield np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def _frames(samples: np.ndarray) -> np.ndarray:
    count = max(0, 1 + (len(samples) - WINDOW) // HOP)
    if count == 0:
        return np.zeros((0, WINDOW), dtype=np.float32)
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    return samples[index]


def features(chunks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 32 ms frame: loudness (dBFS), whistle tone share, whistle frequency."""
    freqs = np.fft.rfftfreq(WINDOW, 1 / RATE)
    band = (freqs >= WHISTLE_BAND[0]) & (freqs <= WHISTLE_BAND[1])
    wide = (freqs >= 300) & (freqs <= 7000)
    hann = np.hanning(WINDOW).astype(np.float32)
    db, tone, pitch = [], [], []
    carry = np.zeros(0, dtype=np.float32)
    for chunk in chunks:
        samples = np.concatenate([carry, chunk])
        frames = _frames(samples)
        used = len(frames) * HOP
        carry = samples[used:]
        if not len(frames):
            continue
        rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
        db.append(20 * np.log10(rms))
        power = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2
        band_power = power[:, band]
        peak = np.argmax(band_power, axis=1)
        # Power within ±2 bins (±31 Hz) of the strongest band frequency.
        rows = np.arange(len(frames))[:, None]
        neighbourhood = np.clip(peak[:, None] + np.arange(-2, 3)[None, :], 0, band.sum() - 1)
        peak_power = band_power[rows, neighbourhood].sum(axis=1)
        tone.append(peak_power / (power[:, wide].sum(axis=1) + 1e-12))
        pitch.append(freqs[band][peak])
    if not db:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(db), np.concatenate(tone), np.concatenate(pitch)


def _runs(mask: np.ndarray, min_len: int, bridge: int) -> list[tuple[int, int]]:
    """(start, end) index runs of True, joining runs separated by up to `bridge` False."""
    runs: list[list[int]] = []
    for i in np.flatnonzero(mask):
        if runs and i - runs[-1][1] <= bridge + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def background(db: np.ndarray, frame_s: float, window_s: float = 30.0) -> np.ndarray:
    """Rolling median loudness over ±window_s/2, computed on one-second blocks."""
    per_block = max(1, round(1.0 / frame_s))
    blocks = len(db) // per_block + (len(db) % per_block > 0)
    padded = np.pad(db, (0, blocks * per_block - len(db)), mode="edge")
    block_db = np.median(padded.reshape(blocks, per_block), axis=1)
    half = max(1, round(window_s / 2))
    # Near the ends, use only the real blocks: repeating the edge would let a
    # cheer at the very end raise its own background.
    smooth = np.array([np.median(block_db[max(0, i - half):i + half + 1])
                       for i in range(blocks)])
    return np.repeat(smooth, per_block)[: len(db)]


def listen(path: str | Path, start_s: float = 0.0, duration_s: float | None = None,
           cheer_rise_db: float = 8.0, cheer_min_s: float = 1.0,
           whistle_share: float = 0.35, whistle_min_s: float = 0.15) -> Listening:
    if not probe(path).has_audio:
        return Listening(has_audio=False)
    db, tone, pitch = features(read_audio(path, start_s, duration_s))
    if len(db) == 0:
        return Listening(has_audio=False)
    frame_s = HOP / RATE
    rise = db - background(db, frame_s)

    def to_time(index: int) -> float:
        return start_s + index * frame_s

    cheers = [
        Sound(to_time(a), to_time(b + 1), float(np.percentile(rise[a:b + 1], 90)))
        for a, b in _runs(rise >= cheer_rise_db, round(cheer_min_s / frame_s),
                          bridge=round(0.3 / frame_s))
    ]
    loud_enough = db >= np.median(db) - 3  # ignore tones buried in silence
    whistles = [
        Sound(to_time(a), to_time(b + 1), float(np.median(tone[a:b + 1])),
              float(np.median(pitch[a:b + 1])))
        for a, b in _runs((tone >= whistle_share) & loud_enough,
                          round(whistle_min_s / frame_s), bridge=2)
    ]
    step = round(0.5 / frame_s)
    loudness = [round(float(np.max(rise[i:i + step])), 1) for i in range(0, len(rise), step)]
    return Listening(has_audio=True, cheers=cheers, whistles=whistles, loudness=loudness)
