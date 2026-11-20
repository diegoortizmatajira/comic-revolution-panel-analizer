# Detector evaluation

Panel detectors are scored against hand-labelled pages, so a change can be judged by numbers
instead of by eye: did it fix the page it targets, and did it break any page that worked?

```sh
uv run python -m app.evaluation --verbose                       # everything available
uv run python -m app.evaluation --corpus tests/fixtures         # committed pages + synthetic only
uv run python -m app.evaluation --detector projection --json out.json
```

By default the corpus is `tests/fixtures` (committed, public-domain), `samples/` (local,
git-ignored, your own pages) and a set of generated pages. Every registered detector is scored
through the same analyzer and reading-order code the API uses.

## What is measured

| Metric | Meaning |
|---|---|
| `prec@50`, `rec@50`, `f1@50` | Panels matched one-to-one at IoU >= 0.5, micro-averaged over all panels |
| `f1@75` | The same at IoU >= 0.75 (tighter boxes) |
| `order` | Share of matched panel pairs that are read in the ground-truth order |
| `page_acc` | Share of pages that are **exactly** right: every panel found, nothing extra, same order |
| `conf_auc` | Chance that a correct page gets a higher `sequence_confidence` than a wrong one. 0.5 means the score carries no information, 1.0 means it separates right from wrong perfectly |

IoU is computed on normalized boxes, which gives the same value as on pixels. Results are also
broken down by tag (`dark`, `borderless`, `spread`, `rtl`, ...) so a weakness shows up as a row
instead of being averaged away.

### Reading the numbers

- `page_acc` is the strictest and the most useful for readers: one missing or merged panel makes
  the page wrong.
- `conf_auc` is currently about 0.5: `sequence_confidence` only measures how cleanly the layout
  could be cut, not whether detection was right, so it does not predict a wrong page. Treat it
  as "layout ambiguity", not "probability of being correct", until that changes.
- The whole-page fallback counts as one predicted panel, so it is correct on a real splash page.

## Scoring an ML model

`ml` and `hybrid` are scored when you give a model (it is never part of the committed baseline):

```sh
uv run python -m app.evaluation --corpus tests/fixtures --ml-model /path/to/panels.onnx --verbose
```

Keep the weights outside the repository (`/models/` is git-ignored). Results for the model tried so
far are in [ADR 0002](adr/0002-optional-ml-detector.md).

## Scoring the edge refiner

`--refine-edges` applies the edge-snapping refiner after each detector. The labels in the corpus are
frame-tight: balloons and captions that hang outside a panel's frame are not part of the label.
The refiner's job is the opposite, to keep those balloons inside the box, so on open-frame pages it
moves boxes away from the labels and strict scores drop. With `projection` on the committed
corpus: f1@50 0.824 to 0.818, f1@75 0.721 to 0.667; `classical` f1@75 0.418 to 0.411. A few pages
(for example `moon-mullings-003-10` and `-17`) grow by 100 to 180 px to include balloons above
the frame.

IoU cannot reward "no content is cropped", so judge the refiner by looking at the boxes, not by
f1@75. A truth-coverage metric (share of each labelled box covered) would be the fair number.

## Label format

A labelled page is an image plus a sibling `<name>.labels.json`:

```json
{
  "version": 1,
  "reading_direction": "ltr",
  "verified": true,
  "tags": ["borderless"],
  "notes": "free text",
  "panels": [
    {"x": 0.05, "y": 0.03, "width": 0.90, "height": 0.30},
    {"x": 0.05, "y": 0.36, "width": 0.43, "height": 0.30}
  ]
}
```

- `panels` are boxes **normalized to the image** (0-1, origin top-left) **in reading order** for
  `reading_direction` (`ltr` or `rtl`).
- A panel is its **frame**. Page numbers, credit lines and captions outside the frame are not
  panels; a caption box inside the frame is part of it. A **title banner** (large lettering above
  the first frame) is labelled as the first panel and tagged `title-banner`: the detector cannot
  tell lettering from a panel that has no frame, and losing a panel costs more than showing a
  title. Where a caption straddles a frame line,
  use the frame line and say so in `notes`. Labels are good to a few pixels, which IoU 0.5
  absorbs.
- Spreads: label every panel of both pages, in reading order across the whole spread.
- Open frames: when a panel has no frame line on a side, the box is the extent of its artwork, and
  balloons or captions floating outside that artwork are left out. Boxes on such pages are only
  accurate to a few tens of pixels; say so in `notes`.
- Unknown keys and boxes outside the page are rejected when the file is loaded.
- `verified: false` marks a draft; the harness ignores drafts unless `--include-unverified`.

Suggested tags: `clean`, `dark`, `noisy`, `aged-paper`, `borderless`, `bleed`, `splash`,
`spread`, `rtl`, `diagonal`, `no-gutter`, `narrow-gutter`, `title-banner`, `captions`,
`circular-inset`, `dense-art`. Generated pages also carry `synthetic`.

## Adding a page

1. Put the image in `tests/fixtures/` (public domain only, with its provenance in
   `tests/fixtures/LICENSES.md`) or in `samples/` (local only; the folder's `.gitignore` keeps
   both the images and their labels out of git).
2. Create a draft: `uv run python -m app.evaluation --propose tests/fixtures/page.jpg`.
   The draft comes from a detector, so **it is biased towards what the detector already does**.
3. Correct every box by looking at the page, set the reading direction and tags, and change
   `"verified"` to `true`.
4. Re-run the evaluation. If the page is in `tests/fixtures`, refresh the baseline (below).

Drafts are only a starting point: never accept one without checking it.

## Baseline and CI

`docs/evaluation/baseline.json` holds the results on the committed corpus (`tests/fixtures`
plus the generated pages). CI runs

```sh
uv run python -m app.evaluation --corpus tests/fixtures --baseline docs/evaluation/baseline.json
```

and fails when `f1@50`, `f1@75`, `order` or `page_acc` of any detector drops by more than
`--max-drop` (default 0.02). Improvements pass. If the corpus itself changed, the comparison is
refused until the baseline is refreshed:

```sh
uv run python -m app.evaluation --corpus tests/fixtures --write-baseline docs/evaluation/baseline.json
```

Commit the refreshed baseline together with the change that caused it. Local `samples/` are not
part of the baseline because they differ between machines.

## Baseline numbers (committed corpus: 23 real pages + 11 generated)

Read the **real** rows. Generated pages are cleaner than scans and flatter the averages.

| detector | pages | f1@50 | f1@75 | order | page_acc | conf_auc |
|---|---|---|---|---|---|---|
| projection, real | 23 | 0.80 | 0.69 | 1.00 | 0.48 | 0.50 |
| projection, synthetic | 11 | 0.97 | 0.92 | 0.96 | 0.64 | 0.50 |
| classical, real | 23 | 0.35 | 0.27 | 0.94 | 0.04 | 0.82 |
| classical, synthetic | 11 | 0.94 | 0.86 | 0.96 | 0.55 | 0.52 |

On real pages `projection` gets 11 of 23 pages exactly right, `classical` 1 of 23. Note that
`classical`'s `sequence_confidence` separates right from wrong pages well (AUC 0.82), because it
flags ambiguous layouts on most pages it gets wrong, while `projection`'s does not (0.50).

Known weaknesses visible in the per-tag rows: pages with no gutter, double-page spreads (panels
of both pages are ordered together, see #5), diagonal gutters for `projection`, and on real
scans: caption text or title lettering in the gutters, thin frames on newsprint, L-shaped panels,
slanted gutters, caption boxes split off from their panel, narrow gutters between panels, and
open frames with balloons overlapping the gutters. Each real page that fails is a strict xfail in
`tests/integration/test_fixture_pages.py`.

## Current limits of the corpus

The corpus is small: 23 labelled real pages and 11 generated pages in the repository (plus
whatever you label locally in `samples/`). Treat the numbers as a regression guard, not as an
accuracy estimate.

- No real manga (RTL) page, double-page spread, full-page splash or dense modern page is
  committed yet; RTL, spreads and splashes are covered only by generated stand-ins.
- All real pages are golden-age or silver-age western comics, which is what public-domain archives
  mostly hold. Fifteen of them come from two issues, so they share a house style.
- Labels were made by one person looking at overlays. For two pages (`public_004`, `public_005`)
  the first draft came from the detector and was then checked by eye, so those labels may lean
  towards what the detector does; the `midnight-mistery-*` and `moon-mullings-*` labels were made
  fully by hand before the detector was run on them.
- Provenance of some pages is incomplete or unverified, see `tests/fixtures/LICENSES.md`.
