# Risks

Ranked within each section, most serious first. Figures marked *illustrative*
are placeholders for the R2 benchmarks, not measurements or quotes.

## Financial

### F1. The Kickstarter rewards can lose money per backer, even before compute

A 12-month Basic reward promises up to 120 analyzed hours. With Kickstarter's
fees (5% platform + about 3% + $0.20 payment processing, using Canadian rates
from Kickstarter's launch; check current rates), and the USD amounts as listed:

| Tier | Pledge | Fees | Net | Net per promised hour | Discount vs 12 × $9.99 |
| --- | --- | --- | --- | --- | --- |
| Super Early Bird | $29 | $2.52 | $26.48 | **$0.22** | 76% |
| Early Bird | $39 | $3.32 | $35.68 | $0.30 | 67% |
| Founding Parent | $49 | $4.12 | $44.88 | $0.37 | 59% |

For comparison, the Basic plan itself nets $0.70–$0.85 per allowance hour
after the App Store's 30% or 15% cut, or about $1.00 with web billing.

What an analyzed hour might cost (*illustrative*):

| Processing speed (GPU time per video hour) | GPU at $0.50/h | $1.00/h | $2.00/h |
| --- | --- | --- | --- |
| 0.25× (4× faster than real time) | $0.12 | $0.25 | $0.50 |
| 0.5× | $0.25 | $0.50 | $1.00 |
| 1× (real time) | $0.50 | $1.00 | $2.00 |

Storage adds more (F3). On these numbers, the $29 tier only breaks even if
the pipeline runs about 4× faster than real time on a cheap GPU, or if most
backers use a small part of their 120 hours. A plan shouldn't rest on that
second hope.

**Mitigation:** the brief's own rule: don't publish rewards until R2/R3 have
measured processing speed on the M4 and on one cloud GPU type. Then model
both worst case (every backer uses all 120 h) and expected use (a season is
roughly 1–2 games a week). Options to discuss, none decided: fewer hours per
month for reward tiers, a shorter term, or a lower cap on the cheapest tier.

### F2. Currency mismatch

The goal is CAD $20,000 but the tiers are in USD. A campaign run from Canada
is priced in CAD, so every tier needs a CAD price. At the current exchange
rate, $29 USD would also be a different number of dollars to a Canadian
backer.

### F3. Storage and upload cost depends on capture resolution and retention

Approximate iPhone HEVC sizes at 30 fps: 1080p ≈ 3.6 GB/hour, 4K ≈ 10 GB/hour,
4K60 ≈ 24 GB/hour. At a standard object-storage list price of about
$0.023/GB-month, keeping one 4K hour for a month costs about $0.23. That
alone matches the Super Early Bird's $0.22 net per hour. Retention (open
question 3) and resolution (Q-new-2) set this number. Uploading 10 GB from a
phone is also slow, and a weak point for reliability.

### F4. App Store rules

Subscriptions sold in the iOS app pay Apple 15–30%. Access bought on
Kickstarter and then unlocked in the app has to fit Apple's rules for content
bought elsewhere. Confirm the current App Review Guidelines (section 3.1)
before promising "redeem in the app".

## Technical

| # | Risk | Mitigation | When |
| --- | --- | --- | --- |
| T1 | **Identity switches when players cross or bunch up.** It's the core of "follow my child", and youth 6v6 is full of crowding | Measure ID switches on labeled clips (R3) before building on them. Combine jersey OCR and re-ID (R5). Manual correction is a feature, not a fallback | R3–R5 |
| T2 | **Players are small and far away** in a wide stationary shot, especially across a full pitch | Detection size is a benchmarked parameter (R1 defaults to 1280 px, not 640). Indoor 6v6 is the easier start | R2 |
| T3 | **Jersey numbers are often unreadable**: motion blur, angle, low gym light | Treat OCR as a confidence signal, not ground truth (brief §4) | R5 |
| T4 | **HDR and variable frame rate** iPhone defaults | Detected and warned at ingest (R1). Analysis resamples to constant rate. HDR needs tone-mapping or a recording guide | R1 ✔, R2 |
| T5 | **Processing speed on the M4 (16 GB)** may be too slow to iterate on full games | Proxies (R1). Benchmark MPS early. Rent a cloud GPU only when justified (ask first: it's a paid service) | R2 |
| T6 | **Highlight detection quality** | Brief: no promises of action recognition before evaluation. Report candidates with confidence and manual review (R7) | R7 |
| T7 | **Long, large uploads from phones** | Resumable uploads; the content-hash asset id makes retries idempotent | R8–R9 |

## Licensing

| # | Risk | Mitigation |
| --- | --- | --- |
| L1 | **Ultralytics YOLO models and code are AGPL-3.0.** Using them in a paid, closed service generally means open-sourcing the service or buying an Ultralytics Enterprise license. Some popular tracker packages that wrap BoT-SORT also carry copyleft licenses | Decide before R2 which detector to build on. Permissively licensed options exist (e.g. Apache-2.0 detectors such as YOLOX or RT-DETR implementations). Check the license of the code *and* of the weights |
| L2 | **Training data licenses**: COCO-pretrained weights, public sports datasets and Roboflow Universe datasets come with mixed or non-commercial terms | Record every dataset and weight file with its license in `docs/` as it's adopted (brief §4) |
| L3 | **Kickstarter's AI policy**: projects that develop AI must disclose their data sources and how consent is handled, and projects without consent safeguards are unlikely to be allowed | See [kickstarter-readiness.md](kickstarter-readiness.md). The data register from L2 doubles as the disclosure |

## Privacy and safety

| # | Risk | Mitigation |
| --- | --- | --- |
| P1 | **Every game video shows other people's children**, not just the subscriber's | Consent model before beta (R9). Possibly blur non-selected players in shared exports. Clear terms on who may record |
| P2 | **Minors' data laws** (e.g. COPPA in the US; PIPEDA and provincial laws in Canada) apply to images of children | Get legal advice before the beta and before the campaign makes privacy claims |
| P3 | **Footage leaking**: through git, logs or shared manifests | R1: footage blocked from git; manifests hold no folder paths. R8: private storage, signed URLs, deletion on a retention schedule |
| P4 | **Evaluation footage** used for R2–R5 | Written permission from each family, stored outside the repo, deletable on request. Never used for training without separate consent |
