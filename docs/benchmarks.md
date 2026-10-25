# Benchmarks

Only measured numbers go here: never estimates. Record the machine, software
versions and the clip each time, so results can be compared across releases.

## How to record a run

- **Machine:** e.g. MacBook Air M4, 16 GB, macOS version
- **Software:** `uv run soccer-recorder doctor` output, plus the git commit
- **Clip:** asset id (never the footage itself), duration, resolution, codec,
  HDR on/off
- **Result:** the number, with units and what was timed

## R1: environment check

| Date | Machine | Result | Notes |
| --- | --- | --- | --- |
| 2026-10-25 | Linux cloud container, 4 vCPU (Python 3.12.3, FFmpeg 6.1.1) | `doctor` passes; 41/41 tests pass in ~4 s | Synthetic clips only |
| — | MacBook Air M4 | *to do* | Run `doctor` and `pytest`; `probe` one real iPhone file |

## R2 onward: what to measure

These feed the reward economics in [risks.md](risks.md#financial). Each one is
needed before the Kickstarter rewards are finalised.

| Measure | Unit | Why |
| --- | --- | --- |
| Detection speed, M4 (MPS) | × real time, per detection size | Feasibility of local iteration |
| Detection + tracking speed, one cloud GPU type | × real time, and $ per analyzed hour | Cost per hour (F1) |
| Ingest time for a full game (hash + copy + proxy) | seconds per video hour | Upload pipeline budget |
| File size per hour at 1080p and 4K, HDR on and off | GB per hour | Storage and upload cost (F3) |
| Detection precision and recall | at IoU 0.5 | R2 acceptance |
| IDF1, HOTA, ID switches per minute | — | R3 acceptance |
| Selected-player coverage | % of on-screen time | R4/R5 acceptance |

## Results

None yet: they need permissioned real footage (see
[gap-analysis.md](gap-analysis.md#gaps-that-need-a-decision-from-you)).
