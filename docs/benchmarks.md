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

## R3/R4: following one player, synthetic games

Not real footage: this checks the logic, not the detector. The scenes
(`tests/synthetic.py`) have players drawn with hair, shirt, shorts and socks,
and teammates in identical kits (only the hair differs). In the *blocker*
scene the followed player walks behind a block for about 2.5 s, longer than
ByteTrack keeps a lost track, while a teammate crosses the other way and a
look-alike plays elsewhere. In *crossing*, an opponent passes in front of
them and a teammate behind. Detections are the true boxes with 4 px noise
and **15% randomly missing**. Run 2026-10-25 (cloud CPU).

| Scene | Followed | Analysed at | On the right player | On someone else | Reconnections |
| --- | --- | --- | --- | --- | --- |
| blocker | target | 15 fps | 83 / 86 visible frames | 0 | 1 (score 0.97) |
| blocker | teammate | 15 fps | 84 / 86 | 1 (last frame) | 1 (0.99) |
| blocker | target | 7.5 fps | 40 / 43 | 0 | 1 (0.98) |
| crossing | target | 15 fps | 115 / 117 | 0 | 1 (0.96) |
| crossing | target | 7.5 fps | 57 / 59 | 1 (last frame) | 1 (0.97) |

The single wrong frames sit at a clip's last frame, where 4 px of noise on a
22 px-wide player is enough to drop the overlap below 0.5. The tests
(`tests/test_follow.py`) require ≥ 90% on the right player and 0 frames on
anyone else, with 3 px noise and 10% missing.

On real footage (to do, M4): `uv run soccer-recorder follow …` on a labelled
clip, compared with the labelled identity.

## R7: events on a synthetic match

`tests/synthetic.py: match_scene` scripts this sequence:

1. the followed player dribbles (0–2 s);
2. passes to a teammate in the same kit (2 s);
3. gets it back (4.3 s);
4. dribbles again;
5. shoots into the right goal (6.5 s);
6. the ball rests in the net (from 6.9 s);
7. the crowd cheers (7.2 s) and the whistle blows (8.5 s, 3.2 kHz).

There are two opponents. Detections are the true boxes with 2 px noise.
Run 2026-10-25.

| Detections missed | Found (time) | Wrong events | Goal |
| --- | --- | --- | --- |
| none | dribble 0.07, pass 2.07, received 4.27, dribble 4.47, shot 6.53, ball in goal 6.87, cheer 7.17, whistle 8.48 | none | 95% (ball + cheering + whistle), by the followed player |
| 15% (players and ball) | the same, within 0.1 s | none | 95%, by the followed player. Event confidences drop to 0.76–0.89 as the ball is seen 83% of the time |
| goals not marked | the same, minus shot and ball in goal | none | *possible goal* 50% (cheering + whistle) |
| no sound track | the same, minus cheer and whistle | none | 55% (ball in the goal only) |

Stats in the clean run: 2 touches, 2 dribbles, 1/1 passes, 1 received,
1 shot, top speed 8.3 km/h (scripted 100 px/s with a 60 px player = 8.4
km/h). Sound alone, on a 20 s test: cheer found at 7.97–11.01 s (true
8–11), whistle at 13.95–14.59 s and 3203 Hz (true 14–14.6 s, 3200 Hz).

## R6: player cam and highlight reel

### Framing, synthetic scenes

The true path of the followed player, rendered at 15 fps. Measured by
`playercam.framing_report`, which every export also saves in
`exports.json`.

- **Centred:** the share of frames with the player's centre in the middle
  60% of the picture, both ways. Frames where the player is at the very edge
  of the whole view, where no crop could centre them, are counted apart.
- **Pan acceleration:** in picture widths per second squared.

Run 2026-10-25.

| Scene | Output | Centred | At the edge | Max pan acceleration |
| --- | --- | --- | --- | --- |
| match | 1280×720 | 100% of 149 | 0 | 0.18 |
| match | 720×1280 | 100% of 149 | 0 | 0.58 |
| crossing | 1280×720 | 100% of 96 | 23 | 0.14 |
| crossing | 720×1280 | 100% of 110 | 9 | 0.46 |
| blocker | 1280×720 | 100% of 111 | 8 | 0.12 |
| blocker | 720×1280 | 100% of 119 | 0 | 0.39 |

Before the camera eased to a stop at the frame's edges, the crossing scene's
vertical cam peaked at 9.3 (a visible jolt where the player walked off the
side). `tests/test_playercam.py` also runs 30 s of a player crossing the
pitch and back, with ±4 px tracking noise and three half-second gaps. It
requires ≥ 95% centred and a peak acceleration ≤ 1.0, in both shapes.

### Export speed, cloud CPU (reference only)

- **Machine:** Linux cloud container, 4 vCPU, no GPU (FFmpeg 6.1.1, software
  decoding and x264), 2026-10-25.
- **Clips:** synthetic `make-sample` pitches at 30 fps.
- **Timed:** `render_playercam`, from decoding to the file with sound.

| Source | Output | Frames | Wall time | × real time |
| --- | --- | --- | --- | --- |
| 1920×1080 H.264, 60 s | 1280×720 | 1,799 | 21.3 s | 2.8 |
| 1920×1080 H.264, 60 s | 720×1280 | 1,799 | 22.5 s | 2.7 |
| 3840×2160 HEVC, 30 s | 1280×720 | 899 | 17.7 s | 1.7 |
| 3840×2160 HEVC, 30 s | 720×1280 | 899 | 18.7 s | 1.6 |

At these speeds, a player cam of a 60-minute 4K game takes about 35 minutes;
a 90 s highlight reel under a minute. On the M4 (to do), try FFmpeg's
VideoToolbox decoding and encoding (risks T8).

## Training, synthetic games

A run of the whole training path to prove it works, not a useful model.
Three synthetic games are labelled from their known truth, one frame every
0.25 s, and given training permission. `dataset` splits them by game:

- train: the match, 40 frames;
- valid: the crossing scene, 32 frames;
- test: the blocker scene, 32 frames, never trained on.

Then `train-detector --size nano --epochs 2 --batch-size 2 --grad-accum 1
--device cpu`. Linux cloud container, 4 vCPU, no GPU, 2026-10-25.

| Model | People: precision | Recall | AP50 |
| --- | --- | --- | --- |
| Standard RF-DETR nano (COCO) | — (nothing above 0.3) | 0.00 | 0.03 |
| Fine-tuned, 2 epochs | 0.84 | 0.69 | 0.64 |

- **Time:** training took 2.1 minutes, about 1 minute per epoch for 40
  small frames. The whole command took 3.3 minutes, with scoring before and
  after.
- **Detection:** the fine-tuned model loads with `--weights`. On a synthetic
  match it finds 4.0 players per frame (there are 4).
- **Why the gain is so large:** the standard model doesn't see the blocky
  synthetic players as people. On real footage it already does, so expect
  much smaller gains there. Only real labelled games can say whether
  fine-tuning helps.

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
