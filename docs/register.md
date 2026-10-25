# Model and data register

Everything the product is built on, with its license and where it came from.
Under [decisions.md](decisions.md) T10, only things that are free for
commercial use go in. This list is also the raw material for the Kickstarter
AI disclosure ([kickstarter-readiness.md](kickstarter-readiness.md)).

Python packages are checked automatically: `tests/test_licenses.py` fails if
any installed package declares a copyleft or non-commercial license. Models
and data can't be checked that way, so they're listed here by hand. Add a row
before anything new is used.

## Models

| Model | Used for | License | Source | Trained on | Notes |
| --- | --- | --- | --- | --- | --- |
| RF-DETR Nano / Small / Medium / Large, COCO-pretrained | R2 detection of people and the ball | **Apache-2.0** (per the [RF-DETR README](https://github.com/roboflow/rf-detr) model table) | Downloaded by the `rfdetr` package on first use from `storage.googleapis.com/rfdetr/`, cached in `~/.roboflow/models/` (366–405 MB each) | COCO, per Roboflow. **To confirm before the AI disclosure:** whether these sizes were also pretrained on Objects365 (the package also lists an Objects365 checkpoint) | Atto/Femto/Pico/XL/2XL are PML 1.0: blocked in code (`rfdetr_detector.SIZES`) |
| DINOv2 backbone (inside RF-DETR) | Image features | Apache-2.0 ([repo](https://github.com/facebookresearch/dinov2)) | Part of the RF-DETR weights | Meta's curated LVD-142M | — |

Fine-tuned detectors (`train-detector`) start from the Apache-2.0 weights
above and learn only from our own permissioned frames. Each one's
`model.json` names its datasheet. They are private, kept under `data/models/`,
and are not listed here one by one.

## Data

| Dataset | Used for | License | Notes |
| --- | --- | --- | --- |
| Synthetic clips (`make-sample`) | Tests | Ours | No real people |
| Our own permissioned youth footage | R2+ evaluation and fine-tuning | Ours, with written consent from every family whose child is identifiable | Planned. Never committed to git |
| SoccerTrack v2 | R3 secondary evaluation, possibly training | CC BY 4.0 data / MIT code ([license](https://raw.githubusercontent.com/AtomScott/SoccerTrack-v2/main/LICENSE-DATA)) | Planned. Credit the authors. CC BY doesn't cover the filmed players' privacy rights |
| SoccerNet, SportsMOT, Roboflow football sets | — | Research-only / non-commercial / unclear | **Not used** (decisions.md T11) |

## Python dependencies worth knowing about

All reviewed as fine for commercial use. Exceptions to the automatic check are
listed in `tests/test_licenses.py` with their reasons.

| Package | License | Why it's listed |
| --- | --- | --- |
| rfdetr | Apache-2.0 | The detector. Its `plus` extra (PML 1.0) is never installed |
| torch, torchvision | torch: BSD-3-Clause, Apache-2.0, MIT, BSD-2-Clause and BSL-1.0 (the Boost license, not the Business Source License). torchvision: BSD | All permissive |
| transformers | Apache-2.0 | Used by RF-DETR to build the backbone |
| supervision | MIT | Detection result types |
| rfdetr[train] extra: pytorch-lightning, torchmetrics, peft, pycocotools, roboflow | Apache-2.0 / BSD | Only with `--extra train`. The `roboflow` package comes with it but is never called: training reads the dataset from disk and uploads nothing |
| faster-coco-eval, torch-hungarian | Apache-2.0 (LICENSE file in the wheel; the metadata omits it) | Training extra; reviewed by hand in `tests/test_licenses.py` |
| certifi, tqdm | MPL-2.0 (tqdm: MPL-2.0 and MIT) | Weak copyleft: only changes to their own files would have to be shared, and we don't change them |
| scipy | BSD | Bundles libgfortran: GPL-3.0 with the GCC Runtime Library Exception, which allows closed-source use |
| matplotlib | PSF-style | Bundles FreeType under "FTL or GPL-2.0": we take the FTL |
| nvidia-* CUDA libraries | NVIDIA proprietary, free to redistribute | Installed on Linux only, as PyTorch's GPU runtime. Not on macOS |
| trackers | Apache-2.0 | ByteTrack, for tracking (R3) |
| opencv-python | Apache-2.0 | Required by `trackers`. Its wheels bundle FFmpeg and Qt as shared LGPL libraries: fine to use as installed; don't modify and redistribute those libraries |
| pillow | MIT-CMU | Draws the name tags |

## Our own analysis code

Ball path, team split, possession, passes, dribbles, shots, goals, cheering
and whistle detection, the player cam and the highlight reel are our own code
(numpy and FFmpeg; Pillow or OpenCV, already listed, to resize and draw). They add no
models, weights or datasets to this register.

## Fonts

Name tags use a font already on the computer that has every letter of the
name (`overlay.FONT_CANDIDATES`: Arial/Arial Unicode/PingFang on macOS,
Liberation/DejaVu/Noto on Linux). The font is only used to draw pixels into
videos on that computer. If nothing suitable is installed, Pillow's built-in
font (a cut-down Aileron Regular, basic Latin only) is used. A hosted service
(R8) should bundle an openly licensed font such as Noto Sans instead of
relying on system fonts.
