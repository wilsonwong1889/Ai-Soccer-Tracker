# AI Soccer Recorder

An iPhone-first soccer recorder for parents: record the whole game from a
tripod, pick your child, get player-focused tracking and highlights. This
folder is the proof of concept, which works on pre-recorded footage.

**Status: past halfway** (R1–R4 and R6 built; R5 and R7 partly built).
Tell it the player's name and jersey number and tap them once. It:

- follows them through the game, with a "Name #10" tag above them;
- works out their analytics: time on the ball, touches, passes, dribbles,
  shots, distance, speed and a heatmap;
- flags goals from the ball going into a marked goal, the crowd cheering and
  the referee's whistle;
- makes videos to keep: a **player cam** (the whole game, zoomed in on your
  child as if someone filmed only them) and a **highlight reel** of their
  best moments, in landscape or vertical for phones.

If it follows the wrong child for a while, tap the right one at that moment
and it follows again.

Every event is an automatic guess with a confidence, which you can tick or
cross. Checked so far on synthetic games and a real two-person clip;
accuracy on real youth footage is still to be measured. See
[the roadmap](docs/roadmap.md) for what each release delivers and how it's
checked.

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
you drop videos in and follow a player (see [the local website](#the-local-website)).

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
- **Follow a player**:
  - Enter their name (optional) and jersey number.
  - Move the slider to a moment when they're clearly visible, then tap them.
  - Choose how much to analyse (the first 1 or 5 minutes, or the whole game)
    and press *Start following*.

  It finds every player, follows yours through the game, and makes a video
  with a "Name #10" tag above them and a ring at their feet. Following a
  second child in the same video reuses the first run's tracking, so it's
  much faster.
- **Detect all players** shows every detected person and the ball, for
  checking detection.
- Progress and time left show under *Activity*, and you can cancel.
- **Watch** any result, or play the original. *numbers* opens the run's
  metrics: how much of the time the player was found, and how many times the
  follower had to reconnect them.
- **Analytics** opens the player's page:
  - **Stat cards:** on the ball, passes, dribbles, shots and goals, distance,
    top speed and sprints, how much their team had the ball, and how often
    the ball was found.
  - **A timeline** with crowd noise, time on the ball, passes, dribbles,
    shots, goals and whistles. Click it to watch that moment.
  - **A heatmap** of where they played.
  - **Moments:** a list of events, each with ▶ to watch and ✓/✗ to say
    whether it really happened.
  - **Mark goals:** drag a box over each goal mouth once per video, so
    "ball in the goal" can be detected. The stats are then re-worked in
    seconds.
  - **Videos to keep:** *Make player cam* or *Make highlight reel*, with
    *Vertical, for phones* if you like. Each appears with **Watch** and
    **Download** (named like `Emma-10-highlights.mp4`). The reel line says
    how many moments it will have. If your ticks and crosses change after a
    reel was made, it's marked *out of date*: make it again.
  - **Wrong player? Fix it:** move the slider to where the follower is on
    the wrong child (the amber box shows who it follows) and tap yours. It
    follows again using all your taps, as a new run beside the old one,
    and keeps your ticks and crosses.

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

## Tracking and following (R3, R4)

```sh
# Track every player; tracks.mp4 shows each track's id
uv run soccer-recorder track <asset id> --duration 120

# Follow one player: their number (and name), and where they are at one moment
uv run soccer-recorder follow <asset id> --number 10 --name Emma --at 12 --point 0.42,0.61
```

`--point` is where the player is at `--at` seconds: video pixels, or
fractions of the frame (0.5,0.5 is the centre). The website gets both from
your tap. How it works:

1. **Tracking:** RF-DETR finds people in 15 frames per second (`--fps`),
   and ByteTrack links them into tracks (`trackers`, Apache-2.0). Each
   tracked box gets a simple colour description of hair, shirt and
   shorts/socks. Saved in a `tracks-…` folder, with the tracks in MOT
   format (`tracks.txt`) for scoring against labelled clips later.
2. **Following:** the tap picks a track. When it ends (the player was
   hidden, or the tracker lost them), the follower picks the track that
   continues it. It must start where the player could have got to in that
   time, at about the same size, and look like the player did when tapped.
   Each reconnection is scored; close calls are marked *uncertain*.
3. **Drawing:** gaps of up to 3 s between pieces are filled in (the tag
   fades while the player is hidden). The path is smoothed and drawn at the
   video's frame rate, so the tag glides. Names with accents or in other
   scripts use a font on your computer that has those letters.

What it can't do yet: tell apart two teammates in the same kit who disappear
together and come out in the same place. Reading the jersey number from the
video (R5) is what will fix that. Until then, the number is shown in the tag
and saved with the run.

## Analytics and events (R5, R7)

`follow` works out the analytics automatically. `analyze` re-works them, for
example after marking the goals:

```sh
uv run soccer-recorder analyze <asset id> --goals 0,0.42,0.05,0.72;0.95,0.42,1,0.72 --height-cm 135
```

How each piece works. All of it is rules over positions and sound, with no
new models or downloads:

- **Ball:** RF-DETR's ball detections, keeping the one that's ball-sized and
  reachable from where the ball just was. Gaps up to 0.5 s are filled.
- **Teams:** players' shirt colours split into two groups. Anyone far from
  both groups (a referee) is "other".
- **On the ball:** the ball within 0.6 body heights of a player's feet. The
  nearest player has it.
- **Passes:** the ball goes from one player to another, at least ~2 m away,
  within 3 s. A teammate receiving it is a *completed* pass; an opponent is
  a *lost* pass. The reverse direction gives *received* and *won the ball*.
- **Dribbles:** at least 1 s on the ball while covering ~2 m.
- **Shots:** the ball leaves your child quickly, heading for a marked goal.
- **Goals:** the ball seen inside a marked goal (55%), plus cheering within
  10 s (+30%), plus a whistle within 20 s (+10%). Cheering alone, loud and
  long, is only a *possible* goal (25–50%): near misses get cheered too.
- **Sound:**
  - *Cheering* is at least 8 dB above the game's running background level
    for a second.
  - A *whistle* is a strong single tone at 2–5 kHz.
- **Distance and speed** are measured in the player's own body heights (which
  roughly cancels perspective), converted with their height (140 cm unless
  you give it). Treat them as estimates.

These rules haven't been measured on real games yet. Your ✓/✗ reviews are
saved in `reviews.json` next to each run and are how they'll be measured.
Known weak spots:

- an indoor hall with several courts has cheering and whistles from the
  other games;
- the ball is small and often hidden in a wide shot (the *Ball seen* card
  shows how often it was found);
- teammates in the same kit can only be told apart by position until jersey
  numbers are read.

## Player cam, highlight reel and fixes (R5, R6)

```sh
# A highlight reel of the latest follow run (or --run <folder name>)
uv run soccer-recorder export <asset id> --kind highlights --vertical
# The whole analysed stretch, zoomed in on the player
uv run soccer-recorder export <asset id> --kind playercam
# The follower got it wrong at 12:40: tap the right child then
uv run soccer-recorder fix <asset id> --at 760 --point 0.42,0.61
```

- **Player cam:** a virtual camera. It crops the original video (up to
  2560 px wide, so 4K keeps its detail) around the player:
  - zoomed so they fill about a third of the picture height, never closer
    than a quarter of the full frame;
  - smoothed both ways in time, since the whole path is known, so it never
    lags;
  - easing out to the whole pitch while they're hidden, and to a gentle stop
    at the edges of the frame.

  The sound is kept. Output is 1280×720 or 720×1280.
- **Highlight reel:** which moments go in:
  - moments you ticked ✓ always do, and moments you crossed ✗ never do;
  - unchecked goals, chances, shots, dribbles, tackles and completed passes
    go in when they're at least 60% sure.

  Each gets a lead-in (6 s before a goal, 4 s before a shot) and a caption
  ("GOAL 12:31"). Overlapping clips merge. The reel is capped at 90 s,
  keeping the best moments, in game order.
- **Fix:** each tap "owns" the part of the game nearest to it, so a new tap
  only changes things around its moment. Taps add up across fixes.
- Every export records in `exports.json` how well it framed the player and
  how smoothly it panned (the R6 checks), and how long it took.

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
