"""Caption claimer: give each panel the lettering that hangs above its top edge.

Western comics put the narration box and the first balloons of a panel above its frame, in the
space between it and the panel above (or the page margin). Detectors return the frame only, so
a reader zoomed to the box would lose that text. For each panel this step looks at the band
between its top edge and the bottom of the panel above it:

* ink that starts within ``touch`` of the top edge is the panel's own lettering;
* the top edge moves up through it until ``gap`` ink-free lines show that it has ended;
* it never moves past the panel above, so text in a gap between two rows belongs to the row
  below it and is claimed once.

Only top edges move, and only upward. The step is opt-in (``PANELS_CLAIM_CAPTIONS``) because
the corpus labels are frame-tight; see docs/evaluation.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from app.domain.models import BoundingBox, DetectedPanel
from app.services.detection.content import contrast_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from app.domain.ports import PanelRefiner


@dataclass(frozen=True, slots=True)
class CaptionClaimer:
    name: str = "caption-claim"
    version: str = "2"  # bump when a change could alter the output (2: skip rows others cover)
    max_expand_ratio: float = 0.12  # farthest a top edge may move, as a fraction of page height
    touch_ratio: float = 0.012  # lettering must start this close to the edge (page height)
    gap_ratio: float = 0.006  # ink-free lines that end the lettering (page height)
    blocked_share: float = 0.5  # a row this covered by other panels ends the claim
    min_overlap: float = 0.3  # share of the panel's width a panel above must cover to bound it
    ink_ratio: float = 0.003  # fraction of a line that must be ink to count
    min_ink_pixels: int = 2
    window: int = 31
    contrast: float = 45.0

    def refine(
        self, page: NDArray[np.uint8], panels: Sequence[DetectedPanel]
    ) -> tuple[DetectedPanel, ...]:
        if not panels:
            return ()
        mask = contrast_mask(page, self.window, self.contrast)
        height = mask.shape[0]
        touch = max(2, round(self.touch_ratio * height))
        gap = max(3, round(self.gap_ratio * height))
        reach = round(self.max_expand_ratio * height)
        tol = max(2, round(self.touch_ratio * height))  # frames that adjoin may overlap a little
        boxes = [(p.bbox.x, p.bbox.y, p.bbox.right, p.bbox.bottom) for p in panels]
        refined = []
        for panel, (x0, y0, x1, y1) in zip(panels, boxes, strict=True):
            floor = min(y0, max(self._ceiling(boxes, (x0, y0, x1, y1), tol), y0 - reach))
            others = [b for b in boxes if b != (x0, y0, x1, y1)]
            top = self._claim(mask, others, (x0, x1, y0), floor, touch, gap)
            refined.append(
                panel
                if top == y0
                else replace(panel, bbox=BoundingBox(x=x0, y=top, width=x1 - x0, height=y1 - top))
            )
        return tuple(refined)

    def _ceiling(
        self, boxes: list[tuple[int, int, int, int]], box: tuple[int, int, int, int], tol: int
    ) -> int:
        """Bottom of the nearest panel above ``box`` (0 when it is on the top row)."""
        x0, y0, x1, _ = box
        bottoms = [
            oy1
            for ox0, _, ox1, oy1 in boxes
            if (ox0, _, ox1, oy1) != box
            and oy1 <= y0 + tol
            and min(x1, ox1) - max(x0, ox0) >= self.min_overlap * (x1 - x0)
        ]
        return max(bottoms, default=0)

    def _claim(  # noqa: PLR0913, PLR0917
        self,
        mask: NDArray[np.bool_],
        others: list[tuple[int, int, int, int]],
        edge_box: tuple[int, int, int],
        floor: int,
        touch: int,
        gap: int,
    ) -> int:
        """Top edge after claiming the unclaimed lettering above ``y0``, never above ``floor``.

        Rows that other panels' boxes cover are not unclaimed: their ink belongs to those panels
        (boxes of diagonal or full-bleed layouts overlap), so the claim stops at the first row
        mostly covered by another panel and ignores the covered part of the rows before it.
        """
        x0, x1, y0 = edge_box
        covered = np.zeros((y0 - floor, x1 - x0), dtype=np.bool_)
        for ox0, oy0, ox1, oy1 in others:
            top, bottom = max(oy0, floor) - floor, min(oy1, y0) - floor
            left, right = max(ox0, x0) - x0, min(ox1, x1) - x0
            if top < bottom and left < right:
                covered[top:bottom, left:right] = True
        blocked = covered.mean(axis=1) > self.blocked_share if covered.size else covered.any(axis=1)
        threshold = max(self.min_ink_pixels, self.ink_ratio * (x1 - x0))
        inked = (mask[floor:y0, x0:x1] & ~covered).sum(axis=1) >= threshold  # rows floor .. y0-1
        edge = None  # index of the last ink row of the claimed run, walking up from y0
        clean = 0
        for index in range(len(inked) - 1, -1, -1):
            if blocked[index]:
                break
            if inked[index]:
                edge, clean = index, 0
            else:
                clean += 1
                if edge is None and clean > touch:
                    return y0  # nothing touches the edge: the gap is a plain gutter
                if edge is not None and clean >= gap:
                    break
        return y0 if edge is None else floor + edge


class ChainedRefiner:
    """Runs refiners one after the other, each on the previous one's boxes."""

    name = "chain"

    def __init__(self, *refiners: PanelRefiner) -> None:
        self._refiners = refiners

    @property
    def version(self) -> str:
        return "+".join(f"{r.name}@{r.version}" for r in self._refiners)

    def refine(
        self, page: NDArray[np.uint8], panels: Sequence[DetectedPanel]
    ) -> tuple[DetectedPanel, ...]:
        result = tuple(panels)
        for refiner in self._refiners:
            result = refiner.refine(page, result)
        return result
