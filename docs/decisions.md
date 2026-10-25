# Decisions and open questions

Product rules come from the product brief (draft, October 2026) and are
copied here unchanged. **Pricing and reward terms change only when you
change them**: anything below marked *proposed* is still yours to confirm.

## Confirmed product rules: Basic plan

| Item | Rule |
| --- | --- |
| Price | $9.99 USD per month (proposed launch price) |
| Monthly allowance | 10 hours = 600 minutes of uploaded AND successfully AI-analyzed footage |
| Recording | Local iPhone recording does not consume allowance; constrained by device storage |
| Usage metering | Deduct actual analyzed source-video duration, not exported highlight duration |
| Failure handling | Refund/reserve-release minutes when our processing fails |
| Renewal | Monthly reset; no rollover initially |
| MVP scope | One selected player per game; personalized highlights and downloads |

Billing requirements (§6): reserve minutes atomically before processing,
settle the actual duration on success, release on failure, make every
operation idempotent, never double-charge a retried job, and show remaining
minutes and per-job history.

## Kickstarter direction: not final

Tentative: 30 days, target CAD $20,000 subject to a costed minimum viable
budget. Backers get digital access, not equity. Founder IDs are permanent and
unique, assigned in an explicitly defined order after pledges are validated.
The public Founder Wall is opt-in.

| Reward idea | Proposed pledge | Limit / benefit |
| --- | --- | --- |
| Super Early Bird | $29 USD | 50 spots; 12 months Basic + Pioneer Founder badge |
| Early Bird | $39 USD | 100 spots; 12 months Basic + Founder badge |
| Founding Parent | $49 USD | 12 months Basic + Founder ID |
| Other tiers | TBD | Family/team plans only after costs and scope are validated |

Not to be published before benchmarking and worst-case fulfilment modelling
(see [risks.md](risks.md#financial) and
[kickstarter-readiness.md](kickstarter-readiness.md)).

## Technical decisions

| # | Decision | Why | Revisit when |
| --- | --- | --- | --- |
| T1 | **Python 3.12** (`.python-version`) | Brief asks for 3.12+. It has the widest support among PyTorch and CV wheels for macOS arm64 | All R2–R5 dependencies support 3.13 |
| T2 | **uv** with a committed `uv.lock` | One command (`uv sync`) reproduces the exact environment, Python included | — |
| T3 | **FFmpeg CLI for all decoding and encoding**, numpy the only Python dependency in R1 | Handles iPhone HEVC, HDR, rotation and VFR the same on macOS and Linux. The same tool renders exports in R6. OpenCV/PyTorch arrive with R2 | Decoding becomes a measured bottleneck (then try PyAV or VideoToolbox) |
| T4 | **Asset id = first 16 hex characters of the file's SHA-256** | Same bytes, same asset, so re-ingesting or retrying never creates a duplicate. R8 jobs and minute metering key on it, which is how "never double-charge" stays true | — |
| T5 | **Original untouched; analysis on downscaled, constant-rate frames** (long side 1280 by default, BGR uint8, upright) | Brief §4: original resolution for exports, small frames for detection. Constant rate gives tracking a steady time step | R2 benchmark picks the real detection size |
| T6 | **Duration from the container** (`ffprobe` format duration, in seconds, kept as float) | It's the "actual analyzed source-video duration" that metering will charge. Rounding is still open (Q-new-1) | Metering design (R8) |
| T7 | **No footage in git, synthetic clips in tests** | Minors' privacy (brief §6). `.gitignore` blocks `data/` and video extensions | — |
| T8 | **Manifests keep the file name, not the folder path** | Paths often contain names | R8 replaces local paths with object-storage keys |
| T9 | **Project in `soccer-recorder/` of this repo, for now** | Only repository available to this session; Rojo ignores the folder | You create a dedicated repo (recommended before R2) |
| T10 | **Decided 2026-10-25 (for now): only code and weights that are free for commercial use (Apache-2.0/MIT/BSD) go into the product.** Stack: RF-DETR (N/S/M/L), `trackers`, gta-link, SAM 2, PARSeq, ViTPose, torchreid ([prior-art.md](prior-art.md)) | Closed-source paid app. Rules out Ultralytics (AGPL), BoxMOT (AGPL), sn-gamestate (GPL), and non-commercial or PML weights | You choose to buy an Ultralytics Enterprise license |
| T11 | **Decided 2026-10-25, follows from T10: no SoccerNet or SportsMOT data, and no checkpoints trained on them** | Research-only / non-commercial terms. Our own footage plus SoccerTrack v2 (CC BY 4.0) instead | Written permission from the dataset owners |
| T12 | **Detection is an optional install** (`uv sync --extra detect`). Defaults: RF-DETR small, score ≥ 0.3, frames downscaled to a 1280 px long side, `--device auto` (Apple GPU, then CUDA, then CPU) | Keeps the core light and its tests torch-free. Small is the middle option until benchmarks say otherwise | R2 benchmarks and `evaluate` results on the M4 |
| T13 | **Every installed package's license is checked by a test** (`tests/test_licenses.py`); models and data go in [register.md](register.md) by hand | Makes T10 hold automatically as dependencies change | — |
| T14 | **The local website (`serve`) uses only Python's standard library** and listens on 127.0.0.1. It refuses other hosts, and changes need a custom header | A one-person tool on the Mac, with no new dependencies to install or license-check. The Host check and the header stop other websites in the same browser from reaching children's footage | R8 builds the real upload/processing API (FastAPI per the brief) |
| T15 | **Following works offline over a whole tracked stretch.** Tap → tracklet → greedy forward/backward linking by predicted position, size and colour similarity; gaps ≤ 3 s filled; Gaussian smoothing | The whole game is recorded, so we can look both ways in time. Every link is saved with its score, so mistakes can be found and corrected | Labelled real clips (R3/R4 numbers) |
| T16 | **A tap is required for now**; the jersey number is shown and saved but not yet read from the video | Reading numbers from a sideline phone is unreliable (one study: 30.6%), and the number-reading model can't be tested in this environment | R5 number reading, tested on the Mac |
| T17 | **Labels use a font already on the computer** that has every letter of the name (Arial Unicode/PingFang on macOS), else Pillow's built-in one | Pillow's built-in font only has basic Latin: "Zoë" came out with a box | A hosted service (R8) ships its own openly licensed font (e.g. Noto Sans) |

## Open questions

From the brief ("clarify before implementation"), unchanged:

1. Does rerunning a game for a different player consume new minutes?
2. Video duration limits.
3. Storage retention.
4. Upload size.
5. Additional-hour pricing.
6. Free tier.
7. Family/team plans.
8. Taxes and app-store billing rules.

Raised during R1:

- **Q-new-1: metering rounding.** Charge per second, or round each job up to
  a whole minute? It's small per game but visible to parents checking their
  usage history.
- **Q-new-2: capture resolution.** 4K gives zoom room for the auto-framed
  export (R6) but about 3× the upload and storage of 1080p (≈10 GB vs ≈3.6
  GB per hour, HEVC 30 fps). This drives cost per hour more than almost
  anything else (see risks F3).
- **Q-new-3: HDR.** Tell parents to turn HDR video off, or tone-map on
  ingest (more compute)?
- **Q-new-4: who gets a Founder ID?** The brief says IDs are assigned to
  validated pledges, but in the reward table only Founding Parent ($49) lists
  one. Do the $29/$39 tiers get one?
- **Q-new-5: when does the "12 months Basic" reward start?** At app launch,
  at redemption, or at a fixed date? It sets how long the liability stays
  open.
- **Q-new-6: reward currency.** A campaign run from Canada is priced in CAD
  (see [kickstarter-readiness.md](kickstarter-readiness.md)), but the tiers
  are listed in USD.
- ~~**Q-new-7: positioning against Trace.**~~ **Resolved 2026-10-25: don't
  raise the price yet.** Basic stays at $9.99/month. Trace PlayerFocus has
  offered per-child highlights since 2023, at about $15–35/month with its own
  camera, so $9.99 is the lowest price found for this feature
  ([prior-art.md](prior-art.md#competitors)). It's worth revisiting only once
  R2–R3 benchmarks give a measured cost per analyzed hour.
