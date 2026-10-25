# Roadmap

The ten releases from the brief, each with a check you can run. Targets marked
*proposed* are starting points to agree on before the release begins. They
can't be validated until real footage exists.

**Where we are (2026-10-25): past halfway.**

- R1–R4 and R6 are built. R5 and R7 are partly built: analytics, manual
  correction, candidate events with confidence and review, and a highlight
  reel.
- Everything works end to end on synthetic games and short real clips.
- The open checks all need real, permissioned youth footage:
  - the speed of each step on the M4;
  - detection accuracy;
  - how long the follower stays on the right child;
  - how often the events are right.

## R1: Repo, environment, ingestion ✔ (pending a Mac run)

- [x] `uv sync` builds the environment from `uv.lock`; `soccer-recorder doctor` passes
- [x] `probe` and `ingest` read H.264 and HEVC, `.mp4` and `.mov`, any rotation;
      warn about HDR, variable frame rate and portrait
- [x] Ingest is idempotent (same file, same asset) and atomic (a failure leaves nothing)
- [x] Original preserved; `iter_frames` gives upright, downscaled, constant-rate BGR frames
- [x] 41 automated tests on synthetic clips; footage can't be committed
- [ ] **On the M4:** `doctor` and `pytest` pass
- [ ] **On a real iPhone file:** `probe` reports a duration within 0.1 s of what
      QuickTime shows; the device, rotation and HDR flags are right

## R2: Detect players

Status: code done; the accuracy check needs labelled real footage.

Deliverable: `soccer-recorder detect <asset>` writes an annotated MP4 and a
metrics JSON.

- [x] `detect`: RF-DETR (Apache-2.0 sizes only, enforced in code) through a video or a
      window of it. Writes `detections.jsonl` (boxes in source pixels), `annotated.mp4`,
      `metrics.json` (speed as × real time, counts, scores, player heights)
- [x] `benchmark`: sizes × input shapes on the same clip, as a table ready for benchmarks.md
- [x] `sample-frames` → label in CVAT ([labeling.md](labeling.md)) → `evaluate`: precision,
      recall, F1, AP50 at IoU 0.5 per class, recall by player height, ignore regions
- [x] Verified end to end on a real image with people (cloud CPU); 169 tests
- [ ] **On the M4:** `benchmark` on a real game clip; pick the default size and shape
- [ ] **Labelled set** (≥ 200 frames, ≥ 2 games) and `evaluate` results against the targets below

- Start with RF-DETR (Apache-2.0 sizes only) and benchmark YOLOX against it
  ([prior-art.md](prior-art.md#proposed-stack-for-r2r5)). No Ultralytics models
- Labeled set: ≥ 200 frames sampled across ≥ 2 games, players and ball boxed
- Reports precision and recall at IoU 0.5, for players and for the ball separately.
  *Proposed:* player recall ≥ 0.90, precision ≥ 0.85
- Reports speed as a fraction of real time on the M4 (MPS) at 2–3 detection sizes,
  and picks the default from that. Goes into benchmarks.md, and feeds risks F1

## R3: Track across frames

Status: built; scoring against labelled real clips is still to do.

- [x] `track`: RF-DETR + ByteTrack (`trackers`, Apache-2.0) at 15 fps, plus a colour
      description of each tracked box. Saves `tracks.npz`, `tracks.txt` (MOT format) and
      `tracks.mp4` with ids. Reused by later follows with the same settings
- [x] Splitting a track where the player's colours change abruptly (an id carried onto an
      opponent)
- [ ] Labelled clips + TrackEval scores (needs footage)

- Roboflow `trackers` (BoT-SORT/ByteTrack), then gta-link offline stitching
- Labeled clips: ≥ 3 × 60 s with identities kept consistent through the whole sequence,
  plus held-out SoccerTrack v2 matches as a secondary check
- Reports IDF1, HOTA and ID switches per minute, scored with TrackEval
- *Proposed:* IDF1 ≥ 0.70 as the baseline to improve on in R5

## R4: Select the child

Status: built. Measured on synthetic games; real footage still to do.

- [x] Website: name + jersey number, then tap the child on a still at any moment;
      `follow` on the command line
- [x] Following through hiding and crossings: reconnects pieces by position, size and
      appearance, scores each link and flags close calls; fills gaps ≤ 3 s; smooths the
      path
- [x] `followed.mp4`: "Name #10" tag above the player and a ring at their feet, at the
      video's frame rate; any script for names
- [x] Synthetic results ([benchmarks.md](benchmarks.md)): 93–98% of visible frames on the
      right player, no frames on another player except single frames at a clip's very end
- [ ] Coverage on real labelled clips

- Pause on a frame, click a player, and the selected track is saved, keyed by
  asset id. It survives a restart
- Try SAM 2 + DAM4SAM to repair ambiguous stretches from that click
- Reports **coverage**: the share of the time the child is on screen during which
  the selected track is actually on them

## R5: Jersey OCR, re-ID, manual correction

Status: started.

- The number is collected, shown and saved.
- A simple colour re-ID and a two-team split by shirt colour are in place.
- Analytics (on/off the ball, distance, speed, sprints, heatmap) are built
  (see R7). Still to do: reading numbers (PARSeq weights are
downloadable; to be built and tested on the Mac), so a player can be found
from the number alone and same-kit teammates told apart.
- [x] Manual correction: *Wrong player? Fix it* on the website shows who is
  followed at any moment; tapping the right child follows again with every
  tap so far (`fix` on the command line). Each tap owns the stretch of game
  nearest to it. The result is a new run, and ticks and crosses carry over.

- Approach: jersey *checking* (PARSeq, Koshkina-style pipeline reimplemented), re-ID
  retrained on our footage, per-game team clustering
- Jersey check accuracy on frames where a number is human-readable
- Re-ID recovers the selected player after an occlusion. Reports the % recovered and
  the median time to recover
- Manual correction: reassign the selected identity from a given frame onward
- *Proposed:* coverage ≥ 0.90 on the evaluation clips without manual correction

## R6: Overlay and auto-framed export

Status: built and measured on synthetic games; real footage and the M4
still to do.

- [x] Player cam: a virtual camera cropped from the original (up to 2560 px),
      with the tag and ring, the game's sound, landscape 1280×720 or vertical
      720×1280. Website buttons with Watch and Download; `export` on the
      command line
- [x] Highlight reel from the reviewed events (see R7)
- [x] Centred: the player's centre in the middle 60% of the picture in 100% of
      frames on the synthetic scenes (frames at the very edge of the pitch view,
      where no crop could centre them, are counted apart). The tests require
      ≥ 95% with tracking noise and gaps
- [x] Smooth: pan acceleration is measured on every export. At most 0.58
      picture widths/s² on the synthetic scenes; the tests cap it at 1.0
      (*proposed*)
- [x] Export speed recorded on every export: 1.6–2.8× real time on a 4-core
      cloud CPU ([benchmarks.md](benchmarks.md#r6-player-cam-and-highlight-reel))
- [ ] **On the M4:** export speed on a real 4K game
- [ ] Framing on real footage

- Rendered from the original resolution with FFmpeg
- The player stays inside the central 60% of the frame in ≥ 95% of tracked frames
  (*proposed*), with no jumpy reframing (camera acceleration capped)
- Reports export speed

## R7: Candidate highlights

Status: candidate events, confidences and review are built; precision on real
games is still to be measured.

- [x] Events for the followed player:
  - on the ball (touches);
  - passes completed, lost and received; ball won;
  - dribbles and shots;
  - ball in a marked goal.
- [x] Goals from combined evidence (ball in goal, cheering, whistle). Cheering
      alone is a *possible* goal.
- [x] Sound: cheering against a rolling background, and referee whistles (2–5 kHz tone).
- [x] Website: stat cards, a clickable timeline, a heatmap, and moments with
      ▶ / ✓ / ✗. Reviews are saved per run (`reviews.json`) and survive
      re-analysis.
- [x] Synthetic match ([benchmarks.md](benchmarks.md)): every scripted event found
      within 0.5 s, the goal at 95% credited to the followed player, also with 15%
      of detections missed
- [x] Highlight reel: ✓ moments always in, ✗ never, the rest when ≥ 60% sure;
      a lead-in per event type and a caption; ≤ 90 s keeping the best; marked
      *out of date* when reviews change
- [ ] Precision per event type from parents' reviews on real games

- Candidates come with a confidence and a reason; the user accepts or rejects each
- Reports precision of the top 10 candidates against human-marked moments
- No "we detect goals/shots" claims until this number exists

## R8: Async processing API

- FastAPI upload and processing endpoints, job queue, object storage with retention
  and deletion
- Minute ledger: reserve before processing, settle on success, release on failure.
  Tested against retries, crashes mid-job and duplicate submissions: never charges
  twice. Remaining minutes and per-job history exposed
- Job id derived from asset id + selected player, so a retried job is the same job

## R9: iPhone app, accounts, beta feedback

- Capture and resumable upload, accounts, parental consent flow, Kickstarter pledge
  import and one-time redemption tokens, Founder IDs (immutable, unique, separate
  from subscription status)

## R10: Limited beta

- Reports tracking quality, job success rate, upload-to-result latency, and
  **measured cost per analyzed hour** against revenue per hour for each plan and reward
