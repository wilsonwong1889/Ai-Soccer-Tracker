# AI Soccer Recorder

An iPhone-first soccer recorder for parents: record the whole game from a
tripod, pick your child, get player-focused tracking and highlights. This
folder is the proof of concept, which works on pre-recorded footage.

**Status: Release 2 of 10** (detection). It finds players and the ball,
with speed and accuracy tooling. The accuracy check still needs labelled real
footage. Nothing tracks or edits yet. See [the roadmap](docs/roadmap.md) for
what each release delivers and how it's checked.

| Doc | What's in it |
| --- | --- |
| [docs/gap-analysis.md](docs/gap-analysis.md) | Audit of the existing repo against the product brief |
| [docs/roadmap.md](docs/roadmap.md) | R1–R10 with measurable acceptance checks |
| [docs/decisions.md](docs/decisions.md) | Confirmed product rules, technical decisions, open questions |
| [docs/risks.md](docs/risks.md) | Technical, financial, privacy and licensing risks |
| [docs/kickstarter-readiness.md](docs/kickstarter-readiness.md) | What has to be true before the campaign goes live |
| [docs/benchmarks.md](docs/benchmarks.md) | How to measure, and the results so far |
| [docs/prior-art.md](docs/prior-art.md) | Existing tools, datasets and competitors, licenses verified: the reuse plan for R2–R5 |
| [docs/labeling.md](docs/labeling.md) | Labelling frames in CVAT and scoring detection (R2) |
| [docs/register.md](docs/register.md) | Every model, dataset and notable package, with its license |

Quickest start: `uv run soccer-recorder serve` opens a local website where
you drop videos in and run detection (see [the local website](#the-local-website)).

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

## The local website

The easiest way to use it:

```sh
uv sync --extra detect          # once (skip --extra detect to only add and play videos)
uv run soccer-recorder serve    # opens http://127.0.0.1:8765 in your browser
```

- **Drop game videos onto the page** (or click to choose). Each one uploads
  and is checked and added to the library. Dropping the same video twice
  doesn't add it twice.
- **Detect players** on any video: the first 1 or 5 minutes, or the whole
  game, with the model size you pick. Progress and time left show under
  *Activity*, and you can cancel.
- **Watch** the result with boxes drawn, or play the original. *numbers*
  opens the run's metrics.

It's a tool for you on your Mac, not the product's upload service (that's
R8). It only answers on this computer: it listens on 127.0.0.1 and ignores
requests from other websites. Uploaded files are moved into the library,
not copied, so a big game doesn't take double the disk. Press Ctrl+C in the
terminal to stop it; running jobs are cancelled. `--port` changes the port
and `--library` the folder.

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

Detection builds on `iter_frames`. The original stays at full resolution for
exports.

## Detection (R2)

Detection needs PyTorch and RF-DETR. They're kept out of the default install
because they're large (several GB on Linux, which pulls in NVIDIA's GPU
libraries; less on macOS):

```sh
uv sync --extra detect
uv run soccer-recorder doctor        # "detect extra" should now be ok, with Apple MPS available
```

On first use, each model size downloads its weights (366–405 MB) from
Roboflow's public storage into `~/.roboflow/models/`. Only the Apache-2.0
sizes can be used (`nano`, `small`, `medium`, `large`); the others are
refused (see [decisions.md](docs/decisions.md) T10).

```sh
# Players and ball through a whole game, or a window of it
uv run soccer-recorder detect <asset id or video> --size small
uv run soccer-recorder detect <asset id> --start 600 --duration 120 --fps 10

# Compare speed across sizes and input shapes on one minute of video
uv run soccer-recorder benchmark <asset id> --sizes nano,small,medium --shapes 704x384

# Accuracy against labelled frames (see docs/labeling.md)
uv run soccer-recorder sample-frames <asset id> --every 15
uv run soccer-recorder evaluate --labels instances_default.json --images data/labeling/<asset id>
```

`detect` uses Apple's GPU automatically (`--device auto`). It writes a run
folder under the asset (`data/library/<id>/runs/detect-…/`) containing:

- `detections.jsonl`: one line per frame, boxes in the original video's
  pixels, `[x1, y1, x2, y2, score, label]`;
- `annotated.mp4`: the frames with boxes drawn (players amber, ball magenta);
  skip it with `--no-video`;
- `metrics.json`: speed, as `realtime_factor` (wall-clock seconds per second
  of video; below 1 is faster than real time), players per frame, how often
  the ball is seen, player heights in pixels, and the detector settings.

RF-DETR's default input is a square (512×512 for small), so a wide 16:9
frame gets squeezed. `--shape 704x384` keeps the proportions at a similar
cost. Whether that finds more small players is exactly what `benchmark` and
`evaluate` are for.

## Tests

```sh
uv run pytest          # a few seconds; generates its own clips
uv run ruff check .
SOCCER_RECORDER_MODEL_TESTS=1 uv run pytest   # also loads real RF-DETR weights
```

Tests that need FFmpeg are skipped (not failed) when it isn't installed.
Detection tests use a stand-in detector, so they pass without the detect
extra. `tests/test_licenses.py` fails if any installed package is copyleft
or non-commercial (decisions.md T10).

## Privacy rules for this folder

The footage is of children. Until consent, retention and access controls
exist (R8/R9):

- **Never commit footage.** `.gitignore` here blocks `data/` and every common
  video extension. Tests generate synthetic clips instead.
- Only use footage you have permission for: from every family whose child is
  identifiable, not only your own.
- Manifests store the original file name but not its folder path. Rename
  files that contain a child's name before ingesting.
