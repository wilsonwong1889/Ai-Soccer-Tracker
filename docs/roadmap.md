# Roadmap

The ten releases from the brief, each with a check you can run. Targets marked
*proposed* are starting points to agree on before the release begins. They
can't be validated until real footage exists.

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

Deliverable: `soccer-recorder detect <asset>` writes an annotated MP4 and a
metrics JSON.

- Detector chosen *after* the license check (risks L1), recorded in decisions.md
- Labeled set: ≥ 200 frames sampled across ≥ 2 games, players and ball boxed
- Reports precision and recall at IoU 0.5, for players and for the ball separately.
  *Proposed:* player recall ≥ 0.90, precision ≥ 0.85
- Reports speed as a fraction of real time on the M4 (MPS) at 2–3 detection sizes,
  and picks the default from that. Goes into benchmarks.md, and feeds risks F1

## R3: Track across frames

- Labeled clips: ≥ 3 × 60 s with identities kept consistent through the whole sequence
- Reports IDF1, HOTA and ID switches per minute
- *Proposed:* IDF1 ≥ 0.70 as the baseline to improve on in R5

## R4: Select the child

- Pause on a frame, click a player, and the selected track is saved, keyed by
  asset id. It survives a restart
- Reports **coverage**: the share of the time the child is on screen during which
  the selected track is actually on them

## R5: Jersey OCR, re-ID, manual correction

- Jersey OCR accuracy on frames where a number is human-readable
- Re-ID recovers the selected player after an occlusion. Reports the % recovered and
  the median time to recover
- Manual correction: reassign the selected identity from a given frame onward
- *Proposed:* coverage ≥ 0.90 on the evaluation clips without manual correction

## R6: Overlay and auto-framed export

- Rendered from the original resolution with FFmpeg
- The player stays inside the central 60% of the frame in ≥ 95% of tracked frames
  (*proposed*), with no jumpy reframing (camera acceleration capped)
- Reports export speed

## R7: Candidate highlights

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
