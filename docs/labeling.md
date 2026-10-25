# Labelling frames for the R2 detection check

R2 passes when detection reaches the targets in [roadmap.md](roadmap.md#r2-detect-players)
on frames *you* have labelled from real, permissioned footage. This page is
the whole loop: sample frames, label them in CVAT, score the detector.

**Footage of children stays on your machine.** Run CVAT yourself (free, the
open-source edition). CVAT's online service is paid, and it would upload
minors' images to a third party: ask before using it.

## 1. Sample frames

```sh
uv run soccer-recorder ingest ~/Movies/game1.MOV
uv run soccer-recorder sample-frames <asset id> --every 15
```

This writes full-resolution JPEGs to `data/labeling/<asset id>/`, named
`<asset id>_00000.jpg`, `_00001.jpg`, … plus `frames.json` (each file's time
in the video). The asset id in the name keeps frames from different games
apart. The roadmap asks for at least **200 frames from at least 2 games**.
One frame every 15 s of a 40-minute game gives 160 per game.

## 2. Run CVAT locally

Needs Docker Desktop. One-time setup, from
[CVAT's install guide](https://docs.cvat.ai/docs/administration/basics/installation/):

```sh
git clone https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Then open http://localhost:8080.

## 3. Label

Create a **project** with exactly these labels (rectangles):

| Label | Draw a box around |
| --- | --- |
| `player` | Every player on the playing surface, both teams, including partly hidden ones (box the visible part) |
| `goalkeeper` | Goalkeepers, if you want them separate (scored the same as players) |
| `referee` | Referees on the playing surface (the detector can't tell them apart from players, so they count as people to find) |
| `ball` | The match ball, whenever any of it is visible |
| `spectator` | People **off** the playing surface: bench, coaches, parents. Detections here are ignored: they count neither for nor against |

Rules that keep the numbers honest:

- **Label every person on the surface in every frame**, however small. Small
  far-away players are exactly what we need to measure.
- A player cut off by the frame edge: label them if you can see at least
  about half of them.
- Two players overlapping: one box each, around what's visible of each.
- Don't label from memory of nearby frames. If you can't see it, don't box it.

CVAT's "AI tools" can propose boxes to speed this up. Check every one.

Create a **task** in the project, upload the JPEGs from step 1, label, then
**Export task dataset → COCO 1.0**. The zip holds
`annotations/instances_default.json`.

## 4. Score

```sh
uv run soccer-recorder evaluate \
  --labels instances_default.json --images data/labeling/<asset id> --size small
```

It prints precision, recall, F1 and AP50 for players and the ball. It also
prints **recall by player height**, which shows whether it's the small,
far-away players that get missed. The full report is saved as JSON next to
the labels. Try `--size nano` / `medium` and a wide `--shape` (e.g.
`704x384`), and record the results in [benchmarks.md](benchmarks.md).

Frames from several games: put all the JPEGs in one folder and label them in
one task (the asset-id prefixes keep the names unique), or evaluate each game
separately.

## Later: tracking labels (R3)

R3 needs the same people followed through continuous clips, with the same
identity throughout. That's labelled in a CVAT *video* task with tracks,
exported in MOT format, and scored with TrackEval. Instructions come with R3.
