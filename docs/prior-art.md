# Prior art: what we can reuse

Research done 2026-10-25 into existing open-source tools, datasets and
competing products for tracking soccer players, reading jersey numbers, and
making per-player highlights.

**How much to trust each line:**

- **✔** means we read the license ourselves from the repository's own LICENSE
  file on 2026-10-25.
- Accuracy figures come from papers and READMEs as reported. We did not
  reproduce them.
- Competitor prices and features come from search-engine extracts, because
  those sites couldn't be opened from the research environment. Re-check them
  before they go into campaign copy.

Licenses decide what a paid, closed-source app can use, so nothing here is
cleared until it's on the "use" side of the tables below.

## Bottom line

1. **Almost every building block exists under Apache-2.0 or MIT.** R2–R5 are
   mostly about assembling and fine-tuning existing parts, not research.
2. **The traps are in pretrained weights and training data, not the code.**
   - Ultralytics YOLO is AGPL, including models you train with it.
   - Many "sports-tuned" checkpoints were trained on SportsMOT (CC BY-NC) or
     SoccerNet (research-only).
   - The best jersey-number pipeline is non-commercial.
   - The sports-specific re-ID models carry an ethical-use license with a
     privacy clause.
3. **No public dataset looks like our footage** (a phone on a tripod, indoor,
   youth players). Our own permissioned footage has to be the main test set
   and the main fine-tuning data. SoccerTrack v2 (static panoramic camera, CC
   BY 4.0) is the best public complement.
4. **Nobody publishes whole-game identity results for one player.** Public
   benchmarks score clips of up to 30 seconds. "Follows your child for the
   whole game" is ours to measure.
5. **The product idea isn't new.** Trace PlayerFocus has sold automatic
   per-child highlights to parents since 2023. The opening is narrower: one
   iPhone, no team sign-up, indoor 6v6, under $10.

## Proposed stack for R2–R5

| Stage | Use | License | Avoid, and why |
| --- | --- | --- | --- |
| Video I/O, drawing | `supervision` | MIT ✔ | — |
| **Detection (R2)** | **RF-DETR Nano/Small/Medium/Large**, fine-tuned on our footage. Input resolution is configurable, which helps with small players. Trains on MPS; Core ML export is experimental. Fallbacks: YOLOX, RT-DETR, D-FINE (COCO-only checkpoints) | Apache-2.0 ✔ (code and these four sizes, per the RF-DETR README) | **Ultralytics YOLO**: AGPL-3.0 ✔, unless we buy their Enterprise license. **RF-DETR Atto/Femto/Pico/XL/2XL**: PML 1.0 ✔. **YOLO-NAS weights**: non-commercial ✔. **DEIMv2**: own license ✔, reported non-commercial |
| **Multi-player tracking (R3)** | Roboflow **`trackers`**: clean-room ByteTrack, BoT-SORT, OC-SORT, McByte | Apache-2.0 ✔ | **BoxMOT**: AGPL-3.0 ✔. **Deep-EIoU**: no license ✔. **StrongSORT**: GPL-3.0 (reported) |
| **Whole-game track stitching (R3/R5)** | **gta-link**: offline split-and-merge of track fragments. It fits because we process a recorded game, not a live stream | MIT ✔ | — |
| **"Click my child" and repair (R4)** | **SAM 2.1** with **DAM4SAM** (memory against look-alikes), or **SAMURAI**; **EdgeTAM** for on-device later | All Apache-2.0 ✔. EdgeTAM README: Core ML export, 16 FPS on iPhone 15 Pro Max ✔ | **SAM 3**: custom "SAM License" ✔. Royalty-free, with no non-commercial clause, but has trade-control terms; needs legal review and a CUDA GPU |
| **Re-ID (R5)** | **torchreid** (OSNet) or **CLIP-ReID** training code, or a **DINOv2** backbone, retrained on our own torso crops. Use per-game hard negatives (other players in the same frame) | MIT ✔ / MIT ✔ / Apache-2.0 ✔ | **PRTReID/BPBReID**: Hippocratic License 3.0 ✔, whose clause 3.1.8 bars privacy interference without written consent. Ambiguous for filming other people's children. Their pretrained person re-ID weights come from datasets with unclear terms (DukeMTMC was withdrawn in 2019) |
| **Jersey numbers (R5)** | **Reimplement** the Koshkina & Elder design: legibility classifier → pose-guided torso crop (**ViTPose**) → **PARSeq** reader → vote across the tracklet. The parent types the child's number, so the job is mostly *checking* "is this a 7?", which is easier than reading any number | PARSeq Apache-2.0 ✔, ViTPose Apache-2.0 ✔ | **jersey-number-pipeline** code and weights: Creative Commons non-commercial ✔. The ideas are reusable; the code and weights are not |
| **Team assignment (R5)** | The Roboflow `sports` team classifier pattern, fitted per game: SigLIP embeddings → UMAP → KMeans on torso crops inside a court area the parent taps once. Show teams as "same kit as my child" | `sports` MIT ✔ (its demo detectors are YOLOv8, AGPL ✔ per its README: don't reuse those) | **sn-gamestate**: GPL-3.0 ✔. Fine for reading, not for shipping |
| **Evaluation** | **TrackEval** (HOTA, IDF1, ID switches) | MIT ✔ | — |
| **Annotation** | **CVAT** community edition, self-hosted, with SAM 2 assistance | MIT ✔ | CVAT Online is paid (≈$23–33/month per seat, reported): ask first |
| Offline labelling help | Vision-language models as *labellers* on a cloud GPU: Florence-2 (MIT), SmolVLM2 (Apache-2.0) | Reported from model cards | PaliGemma (Gemma terms), Moondream 3 (BSL), some Qwen2.5-VL sizes (Qwen license): check before use |

Apache-2.0 components still require shipping their license and NOTICE text
with the app.

## What results to expect

Reported numbers, on professional broadcast footage. Ours will differ.

| Benchmark | Reported | Source |
| --- | --- | --- |
| SportsMOT HOTA (higher is better) | ByteTrack 73.0 ✔, BoT-SORT 73.8, McByte 76.5, Deep-EIoU 77.2, CAMELTrack 80.3, Deep-EIoU + gta-link 81.0 | `trackers` README; papers |
| SoccerNet-Tracking HOTA | ≈84 with perfect detections (ByteTrack 84.0 ✔); 42.4 for the 2023 baseline with real detections | `trackers` README; challenge report |
| SoccerNet Game State Reconstruction (2025) | Baseline ≈22–29 GS-HOTA (the two research reports disagree); winner 63.9 | Challenge reports |
| Jersey numbers, per tracklet | SoccerNet 2023 winner 92.9%; Koshkina & Elder 87.4% | Challenge report; paper |
| Jersey numbers from wide shots | **30.6%** number-reading accuracy (PlayerTV, Norwegian top-flight league) | Paper |
| Re-ID on youth footage | One open project measured whole-player embeddings that could barely tell teammates apart (similarity 0.88–0.95) | reelcut README (self-reported) |

So expect most numbers to be unreadable from a sideline phone. Re-ID, team
colour, motion and the parent's click will carry identity, with jersey reads
as occasional confirmations. That's why manual correction is a core feature.

## Datasets

| Dataset | Camera | What's labelled | Terms | Verdict |
| --- | --- | --- | --- | --- |
| **SoccerTrack v2** (2025) | **Static panoramic 4K**, 10 full university matches | Pitch positions, **jersey-based player IDs**, teams, roles, 12 ball-action classes, a tracking subset | Data **CC BY 4.0** ✔, code MIT ✔; README: "Both permit commercial use" ✔ | **Use** for training and as a secondary test set. Outdoor 11v11 adults, so not a substitute for our own footage |
| TeamTrack | Static fisheye side view + drone; soccer, basketball, **handball** | Boxes + IDs (MOT format) | Repo MIT ✔; no separate data license stated | Get written confirmation first. Handball is a useful indoor proxy |
| TrackID3x3 | **Indoor** fixed camera (3x3 basketball) | Boxes, jersey-based IDs, pose | Repo Apache-2.0 (reported) | Confirm first; indoor proxy |
| SoccerNet (tracking, GSR, jersey, re-ID, actions) | Broadcast | Everything | Research-only per its FAQ (reported); video copyright belongs to the leagues | **Don't use**, even for internal testing, without written permission. Reading their papers and leaderboards is fine |
| SportsMOT | Broadcast | Boxes + IDs | CC BY-NC 4.0 (reported) | **Don't use**. It also rules out checkpoints trained on it |
| Roboflow Universe football sets | Bundesliga broadcast frames | Boxes, pitch keypoints | CC BY 4.0 on the annotations only (reported) | Avoid: the frames themselves aren't covered |
| DFL/IDSSE, SkillCorner open data | No video (positions only) | Tracking + events | CC BY 4.0 / MIT (reported) | Only for event-timing ideas |
| Futsal / indoor / youth tracking | — | — | — | **None found** |

CC BY covers copyright, not the privacy or publicity rights of the people
filmed, so keep consent records for anything we train on.

## Competitors

All figures are from search extracts (see the note at the top). Verify them before quoting.

| Product | What you need | Price (reported) | Per-player output |
| --- | --- | --- | --- |
| **Trace PlayerFocus** | Trace's camera, leased with the subscription | $25–35/month or $180–300/year; team billing $175–200/year per family | **Yes**: a playlist per child, from the jersey numbers on the team roster. Its help centre says it "may miss moments on smaller courts" |
| Veo Go | 2+ iPhones in Veo's rig | ≈$29–55/month (sources disagree) | No. Veo's Player Spotlight isn't on Go and is outdoor-only |
| Veo Cam 3 + Player Spotlight | Veo camera (≈$1,300+) | ≈$67–146/month | Yes, outdoor only |
| Hudl Assist+ | Hudl camera, sold to teams/clubs | Club pricing | Yes, for teams; free parent view |
| XbotGo Chameleon / Falcon | AI phone gimbal ($300) / camera ($699) | No subscription for core features | Not for soccer: its specs say jersey tracking is basketball-only |
| PlayerTrac.ai | Upload only | $14.99–99.99/month | Yes: click or enter a player |
| BallerCam | iPhone case + lens ($249+) | From $29/month | Team-level highlights |

What this means (prices unchanged; this is input for your decisions):

- **Don't claim to be first.** Lead with what's true and different:
  - one iPhone you already own;
  - no team sign-up needed;
  - built for indoor small-sided games, where Veo Spotlight doesn't apply and
    Trace says it is weakest;
  - your own downloads.
- **$9.99/month is the lowest per-player-highlight price found.** Trace and
  PlayerTrac sit around $15. That's room to price higher if the economics in
  [risks.md](risks.md#financial) need it.
- **Crowdfunding in this space has rewarded hardware sold as "no
  subscription".** XbotGo's Falcon campaign raised $2.55M (reported). A
  software subscription is a harder sell, so prepaid, capped founder plans fit
  better than anything "lifetime".
- **XbotGo's marketing implied soccer jersey tracking its specs don't
  support**, and reviews called it out. Show real tracking, failures included.

## Changes to the plan

- **R2**: start with RF-DETR (Apache sizes) and benchmark YOLOX against it.
  Ultralytics is out unless you choose to pay for its license.
- **R3**: `trackers` (BoT-SORT/ByteTrack) plus gta-link stitching, scored with
  TrackEval on our clips and on held-out SoccerTrack v2 matches.
- **R4**: the parent's click seeds the track; SAM 2/DAM4SAM repairs ambiguous
  stretches.
- **R5**: jersey *checking* with PARSeq, re-ID retrained on our footage, and
  per-game team clustering.
- **Data**: own permissioned indoor footage first, SoccerTrack v2 second. No
  SoccerNet or SportsMOT data, and no checkpoints trained on them.
- **Ask before spending**: CVAT Online, an Ultralytics Enterprise license and
  cloud GPUs are all paid. Self-hosted CVAT and the M4 come first.

## Sources

Licenses ✔ were read from these repositories' LICENSE files on GitHub:
[roboflow/sports](https://github.com/roboflow/sports),
[roboflow/supervision](https://github.com/roboflow/supervision),
[roboflow/trackers](https://github.com/roboflow/trackers),
[roboflow/rf-detr](https://github.com/roboflow/rf-detr),
[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics),
[mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot),
[NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT),
[ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack),
[noahcao/OC_SORT](https://github.com/noahcao/OC_SORT),
[hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU),
[tstanczyk95/McByte](https://github.com/tstanczyk95/McByte),
[sjc042/gta-link](https://github.com/sjc042/gta-link),
[TrackingLaboratory/CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack),
[TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab),
[SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate),
[facebookresearch/sam2](https://github.com/facebookresearch/sam2),
[facebookresearch/sam3](https://github.com/facebookresearch/sam3),
[facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM),
[yangchris11/samurai](https://github.com/yangchris11/samurai),
[jovanavidenovic/DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM),
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX),
[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[Peterande/D-FINE](https://github.com/Peterande/D-FINE),
[Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM),
[Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2),
[Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md),
[mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline),
[baudm/parseq](https://github.com/baudm/parseq),
[ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose),
[KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid),
[Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID),
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2),
[VlSomers/prtreid](https://github.com/VlSomers/prtreid),
[VlSomers/bpbreid](https://github.com/VlSomers/bpbreid),
[aarnavshah12/reelcut](https://github.com/aarnavshah12/reelcut),
[JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval),
[cvat-ai/cvat](https://github.com/cvat-ai/cvat),
[AtomScott/SoccerTrack-v2](https://github.com/AtomScott/SoccerTrack-v2),
[AtomScott/TeamTrack](https://github.com/AtomScott/TeamTrack).

Papers and challenge reports:
- [Koshkina & Elder, jersey number recognition (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.html)
- [SoccerNet 2023 challenge results](https://arxiv.org/abs/2309.06006)
- [SoccerNet Game State Reconstruction](https://arxiv.org/abs/2404.11335)
- [McByte (CVPRW 2025)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.html)
- [GTA track stitching](https://arxiv.org/pdf/2411.08216)
- [Deep-EIoU](https://arxiv.org/pdf/2306.13074)
- [PlayerTV](https://arxiv.org/abs/2407.16076)
- [Roboflow: fine-tuned SmolVLM2 for jersey OCR](https://blog.roboflow.com/base-vs-fine-tuned-smolvlm2-ocr/)
- [Apple Vision text recognition](https://developer.apple.com/documentation/vision/recognizetextrequest)

Dataset terms:
- [SoccerNet FAQ](https://www.soccer-net.org/faq)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [SoccerTrack v2 data licence](https://raw.githubusercontent.com/AtomScott/SoccerTrack-v2/main/LICENSE-DATA)
- [IDSSE (Scientific Data 2025)](https://www.nature.com/articles/s41597-025-04505-y)
- [SkillCorner open data](https://github.com/SkillCorner/opendata)

Competitors:
- [Trace PlayerFocus pricing](https://trace.helpcenter.io/content/how-to-subscribe-to-trace-highlights-playerfocus)
- [Trace indoor/futsal FAQ](https://trace.helpcenter.io/content/indoor-soccer-and-futsal-faq)
- [Trace launch (2023)](https://www.businesswire.com/news/home/20230502005375/en/Trace-Launches-PlayerFocus-an-Affordable-Video-Highlights-Service-for-Youth-Soccer)
- [Veo Go plan](https://support.veo.com/hc/en-us/articles/33196190545297-Veo-Go-plan-explained-Features-and-limitations)
- [Veo Player Spotlight](https://shop.veo.co/pages/what-is-veo-player-spotlight)
- [Hudl Assist](https://www.hudl.com/products/assist/soccer)
- [XbotGo Chameleon specs](https://xbotgo.com/pages/xbotgo-chameleon)
- [XbotGo Falcon on Kicktraq](https://www.kicktraq.com/projects/xbotgo/xbotgo-falcon-the-first-team-sports-camera/)
- [PlayerTrac.ai](https://playertrac.ai/)
- [BallerCam](https://ballercam.com)
