# Gap analysis

Audit of the existing setup against the *AI Soccer Recorder* product brief
(draft, October 2026), done before Release 1.

## What exists

- **This repository (`BuildADisaster`) is a Roblox game**: about 100 Luau
  modules synced with Rojo (`default.project.json`), Luau specs in `tests/`,
  and Python scripts in `scripts/` that model the game's economy. None of it
  is related to video, soccer or ML. The soccer project shares nothing with it
  except the git history.
- **None of the account's other repositories is related** either (websites,
  booking apps, other games).
- **No soccer recorder code, environment, footage, labels, benchmarks or
  documentation** existed anywhere before this release.

So the starting point is greenfield. The Roblox game is untouched: everything
for the recorder lives in `soccer-recorder/`, which Rojo doesn't sync.

## Brief vs. current state

| Area (brief section) | Brief asks for | State after R1 |
| --- | --- | --- |
| Repo and environment (§5 R1) | Reproducible Python setup | **Done**: `pyproject.toml`, `uv.lock`, Python 3.12, `doctor` check |
| Sample-video ingestion (§5 R1) | Ingest pre-recorded footage | **Done**: probe, validate, hash, copy or link, optional proxy, manifest |
| Tests and run instructions (§8) | Tests, clear instructions | **Done**: 41 pytest tests, README |
| Decisions, risks, benchmarks in git (§8) | Versioned docs | **Started**: `docs/` (benchmarks empty until R2) |
| Original resolution kept, small frames for detection (§4) | Two resolutions | **Done**: original untouched; `iter_frames` gives downscaled, constant-rate, upright frames |
| Permissioned 5–10 min iPhone footage (§7) | Real test footage | **Missing**: needs you; blocks R2 |
| Detection (§4, R2) | YOLO-style detection | **Built** with RF-DETR (Apache-2.0): `detect`, `benchmark`, `evaluate`. Accuracy unmeasured until real footage is labelled |
| Tracking and selection (§4, R3–R4) | BoT-SORT tracking; parent selects child | **Built:** ByteTrack (`trackers`), tap-to-select with name and number, following through hiding and crossings, "Name #10" overlay. Measured on synthetic games only |
| Jersey OCR, re-ID (§4, R5) | Number recognition, re-ID | Started: number shown/saved, simple colour re-ID, two-team split. Number reading not built |
| Highlights / events (§4, R7) | Candidate events, confidence, manual review | **Built** as transparent rules: passes, dribbles, shots, goals (ball in a marked goal + cheering + whistle), with ✓/✗ review on the website. Precision on real games unmeasured |
| Editing and export (§4, R6) | Auto-framed crops, highlight reels | Missing. The overlay video exists; crops and reels are next |
| Backend (§4, R8) | FastAPI, queue, object storage, Postgres/Supabase | Missing |
| Billing and entitlements (§6) | 600-min reserve/settle/release ledger | Missing; asset ids from R1 give the idempotency key it needs |
| Kickstarter (§3, §6) | Pledge import, redemption tokens, Founder IDs | Missing; reward economics are unvalidated (see risks) |
| Privacy (§6) | Deletion, parental consent, minors' privacy, access control | Only repo hygiene so far (footage can't be committed) |
| Evaluation set (§7) | Identity-consistent labeled sequences | Missing; needs footage plus annotation (CVAT) |
| CI | Not in brief | None; tests run locally. Add once the project has its own repo |

## Gaps that need a decision from you

1. **Where the project lives.** It's in `soccer-recorder/` on the
   `claude/new-session-v5484d` branch of the game's repository, because this
   session only had that repo. It should get its own repository before R2.
   Moving the folder keeps everything working (`git subtree split
   --prefix=soccer-recorder` keeps its history). Creating that repo is your
   call: the brief asks to check before publishing anything.
2. **Real footage.** R2 can't be evaluated on synthetic clips. Needed: 5–10
   minutes of stationary, landscape iPhone footage of a youth 6v6 game, with
   permission from every family whose child is identifiable. Ideally record
   with HDR off.
3. **The open questions in [decisions.md](decisions.md#open-questions)**, of
   which the capture resolution (1080p vs 4K) and the reward currency
   matter soonest.

## Verification gap in R1 itself

R1 was built and tested in a Linux cloud container (Python 3.12.3, FFmpeg
6.1.1). It has **not yet run on the MacBook Air M4**, or on a real iPhone
file. The first thing to do is `uv sync && uv run soccer-recorder doctor &&
uv run pytest` on the Mac, then `probe` one real iPhone recording. Record the
results in [benchmarks.md](benchmarks.md).
