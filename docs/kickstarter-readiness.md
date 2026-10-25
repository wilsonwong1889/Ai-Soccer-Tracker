# Kickstarter readiness

What has to be true before the campaign goes live. It follows the brief's
own order (§7: footage → benchmarks → prototype → parent feedback → *then*
finalise goal, dates, caps, terms and claims), plus the Kickstarter rules
that apply to this project. Nothing here is published or submitted
anywhere.

## Launch gates

| # | Gate | Status | Depends on |
| --- | --- | --- | --- |
| G1 | Pipeline set up and ingesting footage | ✔ R1 (Mac run pending) | — |
| G2 | Permissioned real footage, including consent for **public** use in the campaign video (stricter than consent for internal testing) | Not started | You |
| G3 | A real demo: the child selected, tracked, and an auto-framed clip exported from real footage | Software built (follow, player cam, highlight reel), checked on synthetic games only. Needs G2 footage | G2 |
| G4 | Measured cost per analyzed hour (compute, storage, upload) | Not started | R2–R3 benchmarks |
| G5 | Reward economics modelled worst case (all 120 h used) and expected case, in CAD | Not started | G4, Q-new-6 |
| G6 | 10–20 parents shown real output; willingness-to-pay recorded | Not started | G3 |
| G7 | AI and data disclosure written (below) | Draft points below | Detector/dataset choice (risks L1/L2) |
| G8 | Reward terms settled: Founder ID allocation (Q-new-4), when the 12 months start (Q-new-5), caps, delivery date | Open | G5 |
| G9 | Privacy terms and consent flow reviewed by a lawyer (minors) | Not started | — |

The demo video (G3) is the campaign's centrepiece, and the rules below make
it non-negotiable that it shows real output. So the honest launch date is
after R6, not after R1. R6 is now built, so the demo waits on permissioned
real footage (G2), and on checking the output on it.

## Kickstarter rules that apply

Check each against Kickstarter's current pages before submitting: the
sources are noted, and some are old.

1. **AI disclosure.** Since August 2023, projects that develop AI technology
   must disclose the data they use and how its sources handle consent and
   credit. Kickstarter says it is unlikely to allow a project whose sources
   lack consent safeguards. Undisclosed AI use can get a project suspended.
2. **Honest presentation.** Projects can't mislead or misrepresent. The
   explicit "working prototype, no photorealistic renderings" rule is written
   for hardware and product design, but showing real app output (not
   mock-ups presented as working) is the safe reading for an AI app. Label
   anything that isn't built yet.
3. **Currency and fees.** Campaigns from Canada are in CAD. Fees at
   Kickstarter's Canadian launch were 5% platform (only if funded) plus
   payment processing of 3% + C$0.20 per pledge of C$10 or more. Check
   current pricing. The reward tiers are drafted in USD (risks F2).
4. **No equity or financial returns**: already the plan (digital access only).

## Positioning

From [prior-art.md](prior-art.md#competitors):

- **Don't claim "first".** Trace PlayerFocus has sold automatic per-child
  highlights to parents since 2023. Veo and Hudl have per-player features on
  their own cameras.
- **Claims that are true and different:**
  - one iPhone you already own, no camera to buy;
  - no team sign-up needed;
  - built for indoor small-sided games, where Veo's Player Spotlight doesn't
    apply and Trace says it is weakest;
  - your own downloads.
- **Prove the claims with real footage.** XbotGo's marketing implied soccer
  jersey tracking that its specs say is basketball-only, and reviews called
  it out. Show tracking from kickoff to the final whistle, failures included.
- **Frame rewards as prepaid founder plans with an hours cap**, not
  "lifetime". The successful campaigns in this space sold hardware with "no
  subscription". A subscription reward is a harder sell, and its compute cost
  recurs.

## Draft points for the AI disclosure (G7)

To be finished when R2 picks the detector. For now, these are facts to
gather, not claims to make:

- Which pretrained models are used, what data they were trained on, and
  their licenses (risks L1/L2). Planned: RF-DETR (Apache-2.0, COCO-pretrained),
  SoccerTrack v2 (CC BY 4.0, recorded with participants' consent according to
  its authors), and our own permissioned footage. No SoccerNet or SportsMOT
  data
- Evaluation footage: recorded with written permission from every family
  whose child is identifiable, stored privately, deleted on request
- Whether customer footage is ever used to improve models. Proposed: never,
  without separate opt-in consent
- How training consent is enforced (built): permission is recorded per video
  and per use; training sets are built only from videos with training
  permission, each with a datasheet listing every video used; each model's
  card names the datasheet it learnt from ([training.md](training.md))
- The highlight reel's "learning" is counting the parent's own ticks, on
  their computer: no model is trained on it
- Where footage is stored, who can access it, and when it is deleted

## Rewards: what's drafted and what's unresolved

The tiers stay as drafted in [decisions.md](decisions.md#kickstarter-direction-not-final)
until you change them. The numbers that need to be settled first are in
[risks.md](risks.md#f1-the-kickstarter-rewards-can-lose-money-per-backer-even-before-compute).
In short, the $29 tier nets about $0.22 per promised hour after fees, before
any compute or storage.

Before fulfilment, the code needs (brief §6): import of validated pledges,
one-time redemption tokens, duplicate-claim prevention, an audit trail with
corrections, and Founder IDs that are immutable, unique and separate from
subscription status. These are planned for R9.

## Sources

- [Kickstarter: Introducing our AI policy](https://updates.kickstarter.com/introducing-our-new-ai-policy/)
  and coverage by [TechCrunch](https://techcrunch.com/2023/08/01/kickstarter-requires-generative-ai-projects-to-disclose-additional-info/amp/)
  and [Engadget](https://www.engadget.com/kickstarter-projects-will-soon-have-to-disclose-any-ai-use-145100394.html) (2023)
- [Kickstarter: rules for hardware and product design projects](https://help.kickstarter.com/en-us/articles/16236428-what-are-the-rules-for-hardware-and-product-design-projects)
- Canadian launch, currency and fees: [BetaKit](https://betakit.com/kickstarter-allows-canadians-to-start-creating-projects-today-official-launch-is-september-9th/) (2013)
