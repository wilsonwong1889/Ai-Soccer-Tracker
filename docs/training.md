# Training: teaching the software from your own games

Two kinds of learning are built in. Neither sends anything off your computer.

1. **The highlight reel learns from your ticks and crosses.** No AI model is
   trained: it counts how often each kind of moment was right, and from
   that decides how sure the software must be before a moment goes in the
   reel. It works as soon as you tick moments on the Analytics page.
2. **The player detector can be fine-tuned** on frames you have labelled
   from your own games. This one is real AI training. It needs permission
   from the families, labelled frames, and some hours of computer time.

## 1. Learning from ticks and crosses

On the Analytics page, every ✓ or ✗ is saved and counted across all your
games. Once you've checked **10 moments of one kind** (say, dribbles), the
software works out the lowest confidence at which at least 80% of them were
right:

- the highlight reel then takes that kind of moment only from that
  confidence up;
- a kind that's wrong too often goes in only when you tick it;
- moments you tick always go in, and moments you cross never do.

The Moments list shows what has been learnt ("Learnt from your checks:
dribble right 83% → reel takes them from 75% sure"). On the command line:

```sh
uv run soccer-recorder reviews
```

prints the table and saves it as `calibration.json` in the library folder.
These counts are also the R7 measurement: precision per kind of moment, on
real games.

## 2. Fine-tuning the detector

Only frames from videos whose families agreed to **training** can be used.
The software enforces this; there's no way around it.

### Step 1: record permission

On the website, each video has three boxes under "Families agreed to":
*measuring accuracy*, *training our AI* and *showing publicly*. Or:

```sh
uv run soccer-recorder consent <asset id> --training \
  --note "signed forms from all 12 families, folder 'Team 2026'"
```

Tick a box only if **every family whose child can be seen** agreed to that
use, in writing. Keep the forms outside this folder. The note says where
they are; never put children's names in it. Permission for testing doesn't
cover training (risks P4).

### Step 2: sample and label frames

Follow [labeling.md](labeling.md): `sample-frames` (the asset id in each
file name is how permission is checked later), label in CVAT on your own
computer, and export **COCO 1.0**. For training, label **every person you
can see**, spectators included: an unlabelled person teaches the model that
people there aren't people.

How many: start with **200 frames from at least 3 games**. With 3 or more
games, whole games are kept back for testing, which is the honest test.

### Step 3: build the training set

```sh
uv run soccer-recorder dataset --name indoor-2026-11 \
  --labels game1/instances_default.json --images data/labeling/<asset id 1> \
  --labels game2/instances_default.json --images data/labeling/<asset id 2> \
  --labels game3/instances_default.json --images data/labeling/<asset id 3>
```

It writes `data/datasets/indoor-2026-11/` with `train/`, `valid/` and
`test/` folders. It also writes `datasheet.md`, which lists every video
used, when its permission was recorded, and what was left out and why. That
datasheet is the raw material for the Kickstarter AI disclosure. Datasets
are never overwritten, so each model stays traceable to what it learnt from.

### Step 4: train

```sh
uv sync --extra detect --extra train      # once
uv run soccer-recorder train-detector data/datasets/indoor-2026-11 --name indoor-v1
```

It does three things:

1. scores the standard model on the test frames;
2. trains (20 epochs of RF-DETR small by default, on the Apple GPU when
   available);
3. scores the new model on the same frames.

It then says whether the new model is **better**: player AP50 up, and the
ball not worse by more than 0.02. Everything is in
`data/models/indoor-v1/model.json`, the model card:

- what it started from;
- the datasheet it learnt from;
- the settings and time taken;
- the before and after scores.

How long: on this project's cloud test machine (4 CPU cores, no GPU),
RF-DETR nano took about 1.5 minutes per epoch on 40 small frames. Real
1080p/4K frames and the small size take much longer. Measure it on the M4
first (benchmarks.md), and ask before renting a cloud GPU: it costs money.

### Step 5: use it, if it's better

```sh
uv run soccer-recorder serve --weights data/models/indoor-v1
uv run soccer-recorder follow <asset id> --weights data/models/indoor-v1 ...
```

`--weights` works for `detect`, `track`, `follow`, `evaluate` and `serve`.
Runs made with a fine-tuned model are named after it, so its cached tracking
never mixes with the standard model's. Only models with a model card load,
so a model whose training data isn't known can't be used.

## Rules

- **Trained weights stay private.** A model trained on children's footage
  is derived from it. It lives under `data/` (never committed), and isn't
  shared without the families' permission.
- **Customer footage is never used for training** without separate, opt-in
  permission (kickstarter-readiness.md, G7).
- **Only our own permissioned footage, and openly licensed data** (e.g.
  SoccerTrack v2, CC BY 4.0) once it's added with its own datasheet entry.
  No SoccerNet or SportsMOT (decisions.md T11).
- If a family withdraws permission: untick *training our AI* on their videos,
  rebuild the training set without them, and retrain. Delete the models
  trained on the old set (their model cards list the videos they used).
