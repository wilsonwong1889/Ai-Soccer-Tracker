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

## R2: detection speed, cloud CPU (reference only)

Not representative of the M4, which has a GPU. This only shows the
`benchmark` command working and the relative cost of each size.

- **Machine:** Linux cloud container, 4 vCPU, no GPU (torch 2.14.1, rfdetr 1.11.2), 2026-10-25
- **Clip:** 3 s, 640×480, 10 fps, made from a still photo of two people and a ball
  (OpenCV sample image, not committed)

| Detector | Input | Device | × real time (at 10 fps) | Detect p50 ms | Players/frame | Ball seen |
| --- | --- | --- | --- | --- | --- | --- |
| rfdetr-nano | 384x384 | cpu | 1.4 | 131.7 | 2.0 | 100% |
| rfdetr-nano | 640x480 | cpu | 2.677 | 254.1 | 2.0 | 100% |
| rfdetr-small | 512x512 | cpu | 2.218 | 216.2 | 2.0 | 100% |
| rfdetr-small | 640x480 | cpu | 2.681 | 258.7 | 2.0 | 100% |
| rfdetr-medium | 576x576 | cpu | 2.844 | 272.0 | 2.0 | 100% |
| rfdetr-medium | 640x480 | cpu | 2.763 | 258.0 | 2.0 | 100% |

On this CPU, analysing every frame of 30 fps video would take about 4–8× real
time. The M4 numbers decide feasibility.

**To do on the M4:** `uv run soccer-recorder benchmark <asset> --sizes nano,small,medium --shapes 704x384`
on a real game clip, and paste the table here.

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
