# AI Soccer Recorder

An iPhone-first soccer recorder for parents: record the whole game from a
tripod, pick your child, get player-focused tracking and highlights. This
folder is the proof of concept, which works on pre-recorded footage.

**Status: Release 1 of 10** (ingestion and setup). Nothing here detects,
tracks or edits yet. See [the roadmap](docs/roadmap.md) for what each release
delivers and how it's checked.

| Doc | What's in it |
| --- | --- |
| [docs/gap-analysis.md](docs/gap-analysis.md) | Audit of the existing repo against the product brief |
| [docs/roadmap.md](docs/roadmap.md) | R1–R10 with measurable acceptance checks |
| [docs/decisions.md](docs/decisions.md) | Confirmed product rules, technical decisions, open questions |
| [docs/risks.md](docs/risks.md) | Technical, financial, privacy and licensing risks |
| [docs/kickstarter-readiness.md](docs/kickstarter-readiness.md) | What has to be true before the campaign goes live |
| [docs/benchmarks.md](docs/benchmarks.md) | How to measure, and the results so far |

## Setup (macOS, Apple Silicon)

You need [Homebrew](https://brew.sh). Then, from this folder:

```sh
brew install ffmpeg uv     # FFmpeg decodes iPhone video; uv manages Python
uv sync                    # installs Python 3.12 and the locked dependencies into .venv
uv run soccer-recorder doctor
```

`doctor` checks Python, FFmpeg, the H.264/HEVC decoders and the library
folder, and exits non-zero if anything required is missing. It ends with
`All required checks passed.` when the machine is ready.

Without uv: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e . pytest`
(unpinned, so not exactly reproducible). Linux works the same with FFmpeg from
your package manager.

## Usage

```sh
# Look at a video without changing anything
uv run soccer-recorder probe ~/Movies/IMG_4821.MOV

# Add it to the library (./data/library): validates, hashes, copies
uv run soccer-recorder ingest ~/Movies/IMG_4821.MOV

# Symlink instead of copying (a full game is several GB), and write a
# 1280px constant-frame-rate proxy for faster detection experiments
uv run soccer-recorder ingest ~/Movies/IMG_4821.MOV --link --proxy

uv run soccer-recorder list
```

Ingesting is safe to repeat: each video is stored under the first 16
characters of its SHA-256, so the same file always maps to the same asset
and a second ingest reports `Already in library`. A failed or interrupted
ingest leaves nothing behind. Set `SOCCER_RECORDER_LIBRARY` or pass
`--library` to keep the library elsewhere, such as an external drive.

```
data/library/3f9a1c0e5b7d2a64/
  manifest.json   duration, size, codec, rotation, HDR/VFR flags, device, hash
  source.mov      the untouched original (or a symlink to it)
  proxy.mp4       only with --proxy
```

`probe` and `ingest` warn about things that will matter later:

- **HDR** (iPhones record it by default): detection frames look washed out.
  For test recordings, turn off *Settings > Camera > Record Video > HDR Video*.
- **Variable frame rate** (common indoors in low light): frames are resampled
  to a constant rate for analysis.
- **Portrait**: a sideline tripod recording should be landscape.

### No footage yet? Make a synthetic clip

```sh
uv run soccer-recorder make-sample data/sample.mov            # 10 s, 1080p HEVC
uv run soccer-recorder make-sample data/portrait.mov --rotation 90
```

This makes a green pitch with coloured boxes moving like players. It exercises
the plumbing only: it says nothing about detection quality.

### In Python

```python
from soccer_recorder.frames import iter_frames
from soccer_recorder.library import ingest

asset, created = ingest("IMG_4821.MOV")
for frame in iter_frames(asset.source_path, max_side=1280, fps=10):
    frame.image   # upright BGR uint8 array, long side <= 1280 px
    frame.time_s  # seconds from the start of the video
```

R2 detection will build on `iter_frames`. The original stays at full
resolution for exports.

## Tests

```sh
uv run pytest          # 41 tests, a few seconds; generates its own clips
uv run ruff check .
```

Tests that need FFmpeg are skipped (not failed) when it isn't installed.

## Privacy rules for this folder

The footage is of children. Until consent, retention and access controls
exist (R8/R9):

- **Never commit footage.** `.gitignore` here blocks `data/` and every common
  video extension. Tests generate synthetic clips instead.
- Only use footage you have permission for: from every family whose child is
  identifiable, not only your own.
- Manifests store the original file name but not its folder path. Rename
  files that contain a child's name before ingesting.
