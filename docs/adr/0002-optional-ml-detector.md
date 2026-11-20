# ADR 0002: Optional ML detector, bring your own weights

- Status: accepted
- Context: issue #4; depends on the evaluation harness (#2)

## Decision

- The service can run a YOLO-style **ONNX** panel model through `onnxruntime`, alone (`ml`) or as a
  fallback to the classical detector (`hybrid`). It is selected with `PANELS_DETECTOR`.
- **No weights ship with this project.** The deployer supplies a model file
  (`PANELS_ML_MODEL=/path/to/model.onnx`) and is responsible for its licence. The Docker image
  never contains weights: mount them. `onnxruntime` is an optional extra (`uv sync --extra ml`,
  `docker build --build-arg WITH_ML=true`), so the default install is unchanged.
- The model's SHA-256 (first 12 hex digits) is part of the response `version`, so results from
  different weights never share a cache entry.
- `PanelDetector.detect` now returns `Detection(panels, method)`, so a composite detector reports
  which component actually answered each page.

## Why not ship a model

No candidate has a clean licence chain (checked against the model cards on 2026-11-19; nothing
was verified beyond what the cards say):

| Candidate | Weights licence | Training data | Notes |
|---|---|---|---|
| [Inkwell panel detector](https://huggingface.co/cedarrapidsboy/inkwell-panel-models) | AGPL-3.0, including network use | ~3,800 annotated pages; origin and licence not stated | YOLOv8-nano, 640 px, ships `best.onnx`, reported mAP50 0.977 |
| [mosesb YOLOv12x](https://huggingface.co/mosesb/best-comic-panel-detection) | card says Apache-2.0 | custom Roboflow set; licence not stated | extra-large model; the Ultralytics code behind it is AGPL, so the claim is doubtful |
| [Oliverdsfdsf YOLO26n-seg](https://huggingface.co/Oliverdsfdsf/comic-panels-text-detect) | card says Apache-2.0 | not stated | targets comics, manga and manhwa; no metrics |
| [DeepPanel](https://github.com/smather/DeepPanel) | code Apache-2.0, weights unlicensed | private, 550 pages | TensorFlow U-Net |

This repository is MIT. Shipping or requiring an AGPL model would change what hosting the service
obliges you to publish. Training our own needs a licensable annotated dataset we do not have (the
corpus has 34 pages). So the project provides the mechanism and leaves the choice of weights, and
its licence, to the deployer.

## Measurements

All numbers are from the evaluation harness (`docs/evaluation.md`) on the 34-page corpus (23 real
pages, 11 generated), using the Inkwell `best.onnx` (`1:f240e1296efd`) **for local measurement
only: it was not committed or distributed**. "Pages exactly right" is the harness `page_acc`.

| Detector | Real pages exactly right | Real F1@50 | Generated pages exactly right |
|---|---|---|---|
| `projection` | 11 of 23 | 0.80 | 7 of 11 |
| `ml` | 19 of 23 | 0.97 | 7 of 11 |
| `hybrid` | 18 of 23 | 0.96 | 8 of 11 |

- **Hybrid never does worse than `projection` on any of the 34 pages**, and fixes 8 it got wrong.
- **`ml` alone is better on real pages than `hybrid`** (19 against 18): the hybrid cannot tell when
  a plausible-looking result is wrong (three `projection` failures merge panels but look normal).
  Hybrid is the conservative option: it keeps pages `projection` already gets right, and runs the
  model on only 9 of the 23 real pages (39%).
- Reading order matters as much as detection. With the old strict XY-cut the same model got only
  10 of 23 real pages exactly right even though it found nearly every panel, because its boxes
  overlap by a few pixels and no clean cut exists. The XY-cut now has a **gap tolerance** of 8% of
  the group's extent (ordering version 2). Against ML boxes the result rises to 19 of 23 and plateaus
  from 8% to 12%; `projection` is unaffected.

### Hybrid policy

`projection` is distrusted, and the model asked, when its result has **fewer than 3 panels** or a
panel whose long side is **at least 5 times its short side** (a caption strip). A thin panel that
lies wholly above every other panel is the title banner `projection` returns on purpose as a first
panel, so it does not count: the model never returns titles and would only drop it.
On the corpus these rules flag 9 of the 12 pages `projection` gets wrong and none of the 11 it
gets right; the model's answer replaces `projection`'s when it finds at least one panel.

The three thresholds (3 panels, 5:1, 8% tolerance) were chosen on this small corpus and are
untuned beyond it. Revisit them when the corpus grows.

### Cost (this machine, 24 cores; pinning to one core gave the same figures)

| | |
|---|---|
| Model file | 12.3 MB (mounted, not in the image) |
| Inference | median 15 ms per ~975x1400 page (`projection`: 37 ms) |
| Cold start | 0.27 s in process (import `onnxruntime`, hash the file, load, first inference); container ready 0.6 s |
| Docker image | 429 MB without the extra, 513 MB with `WITH_ML=true` (+84 MB) |

## Caveats

- **Possible test-set contamination.** The Inkwell training pages are undocumented. If they
  include golden-age scans like ours, its numbers here are optimistic.
- **AGPL.** Measuring the Inkwell weights locally is not distribution. Deploying them behind a
  service others use is a different matter; read its licence and decide before doing so.
- **Small corpus.** 23 real pages, all western comics, 15 of them from two issues.
- The model must be YOLO-style single-class (or multi-class with `PANELS_ML_PANEL_CLASS`) and
  take a square input; `PANELS_ML_OUTPUT_FORMAT=end2end` selects the `(N, 6)` NMS-free layout. Other
  architectures need another decoder.
