"""Gutter-based panel detector using projection profiles (recursive XY-cut on pixels).

1. Estimate the page background colour from a thin strip around the page edge.
2. Mark every pixel whose colour differs from it as "content".
3. Recursively split a region wherever a run of rows (or columns) contains almost no
   content: that run is a gutter. Rows are tried first, then columns, then each piece
   is trimmed to its content and split again. Regions that cannot be split are panels.

Unlike contour detection this does not need panels to be disconnected blobs, so panels
whose borders or art nearly touch are still separated as long as a gutter runs between
them. When no straight gutter exists, a slanted one is followed with a minimum-content path
(seam). Known limitations: panels with no gutter at all stay merged; title banners are
not told apart from panels; borderless panels
whose artwork contains background-coloured bands can be split by mistake; a full-bleed
splash page (no margin to sample) yields no panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import BoundingBox, DetectedPanel, Detection
from app.services.detection.content import content_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Mask = NDArray[np.bool_]

_Region = tuple[int, int, int, int]  # y0, y1, x0, x1 (end-exclusive)


class ProjectionPanelDetector:
    name = "projection"
    version = "2"

    def __init__(  # noqa: PLR0913
        self,
        *,
        color_tolerance: float = 40.0,
        gutter_content_ratio: float = 0.005,
        gutter_relative_threshold: float = 0.2,
        min_gutter_ratio: float = 0.0015,
        min_area_ratio: float = 0.02,
        border_fraction: float = 0.01,
        max_depth: int = 12,
        seam_content_ratio: float = 0.004,
        min_seam_side_ratio: float = 0.03,
    ) -> None:
        self._tolerance = color_tolerance
        self._gutter_ratio = gutter_content_ratio
        self._gutter_relative = gutter_relative_threshold
        self._min_gutter_ratio = min_gutter_ratio
        self._min_area_ratio = min_area_ratio
        self._border_fraction = border_fraction
        self._max_depth = max_depth
        self._seam_content_ratio = seam_content_ratio
        self._min_seam_side_ratio = min_seam_side_ratio

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        height, width = page.shape[:2]
        mask = self._content_mask(page)
        regions: list[_Region] = []
        self._split(mask, (0, height, 0, width), 0, regions)

        min_area = self._min_area_ratio * width * height
        panels: list[DetectedPanel] = []
        for y0, y1, x0, x1 in regions:
            if (y1 - y0) * (x1 - x0) < min_area:
                continue
            fill = float(mask[y0:y1, x0:x1].mean())
            panels.append(
                DetectedPanel(
                    bbox=BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                    confidence=min(1.0, max(0.0, fill)),
                )
            )
        return Detection(panels=tuple(panels), method=self.name)

    def _content_mask(self, page: NDArray[np.uint8]) -> Mask:
        return content_mask(page, self._tolerance, self._border_fraction)

    def _split(self, mask: Mask, region: _Region, depth: int, out: list[_Region]) -> None:
        y0, y1, x0, x1 = region
        sub = mask[y0:y1, x0:x1]

        rows = np.flatnonzero(sub.mean(axis=1) > self._gutter_ratio)
        if rows.size == 0:
            return
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        cols = np.flatnonzero(sub[top:bottom].mean(axis=0) > self._gutter_ratio)
        if cols.size == 0:
            return
        left, right = int(cols[0]), int(cols[-1]) + 1

        y0, y1, x0, x1 = y0 + top, y0 + bottom, x0 + left, x0 + right
        sub = mask[y0:y1, x0:x1]

        if depth < self._max_depth:
            for axis in (1, 0):  # rows first (horizontal gutters), then columns
                profile = sub.mean(axis=axis)
                segments = self._segments(
                    profile, self._threshold(profile), self._min_gutter(len(profile))
                )
                if len(segments) > 1:
                    for s0, s1 in segments:
                        child = (
                            (y0 + s0, y0 + s1, x0, x1) if axis == 1 else (y0, y1, x0 + s0, x0 + s1)
                        )
                        self._split(mask, child, depth + 1, out)
                    return
        if depth < self._max_depth:
            halves = self._seam_split(sub)
            if halves is not None:
                for half in halves:
                    padded = np.zeros_like(mask)
                    padded[y0:y1, x0:x1] = half
                    self._split(padded, (y0, y1, x0, x1), depth + 1, out)
                return
        out.append((y0, y1, x0, x1))

    def _seam_split(self, sub: Mask) -> tuple[Mask, Mask] | None:
        """Cut a region along a slanted gutter: the cheapest path through the content mask.

        Used only when no straight row/column gutter exists. The path may drift one pixel per
        step, so it follows a slanted gutter; it is accepted only if it runs through (almost)
        no content for the whole region, i.e. it really is a gutter and not a cut through art.
        """
        best: tuple[float, Mask, Mask] | None = None
        for transpose in (False, True):  # False: seam runs left to right, cutting rows apart
            oriented = sub.T if transpose else sub
            found = self._cheapest_seam(oriented)
            if found is None:
                continue
            cost, upper, lower = found
            if best is None or cost < best[0]:
                best = (cost, upper.T if transpose else upper, lower.T if transpose else lower)
        return None if best is None else (best[1], best[2])

    def _cheapest_seam(self, sub: Mask) -> tuple[float, Mask, Mask] | None:
        height, width = sub.shape
        clearance = self._min_gutter(height) | 1
        blocked = cv2.dilate(sub.astype(np.uint8), np.ones((clearance, 1), np.uint8)) > 0
        cost = blocked.astype(np.float32)
        total = np.zeros((height, width), dtype=np.float32)
        step = np.zeros((height, width), dtype=np.int8)
        total[:, 0] = cost[:, 0]
        inf = np.float32(np.inf)
        for x in range(1, width):
            prev = total[:, x - 1]
            up = np.concatenate(([inf], prev[:-1]))
            down = np.concatenate((prev[1:], [inf]))
            # Prefer going straight on ties, so a straight gutter stays straight.
            stacked = np.stack((prev, up + 1e-3, down + 1e-3))
            choice = stacked.argmin(axis=0)
            total[:, x] = stacked.min(axis=0) + cost[:, x]
            step[:, x] = choice
        end = int(total[:, -1].argmin())
        if total[end, -1] > self._seam_content_ratio * width * 10:  # blocked pixels allowed
            return None
        path = np.empty(width, dtype=np.int64)
        y = end
        for x in range(width - 1, -1, -1):
            path[x] = y
            c = step[y, x]
            y += 0 if c == 0 else (-1 if c == 1 else 1)
        if float(blocked[path, np.arange(width)].mean()) > self._seam_content_ratio:
            return None
        rows = np.arange(height)[:, None]
        upper = sub & (rows < path[None, :])
        lower = sub & (rows > path[None, :])
        area = sub.sum()
        floor = self._min_seam_side_ratio * area
        if upper.sum() < floor or lower.sum() < floor:
            return None
        return float(blocked[path, np.arange(width)].mean()), upper, lower

    def _min_gutter(self, length: int) -> int:
        return max(2, round(length * self._min_gutter_ratio))

    def _threshold(self, profile: NDArray[np.float64]) -> float:
        """Content fraction at or below which a row/column counts as gutter.

        Relative to how full the region's rows/columns normally are, because hand-drawn
        or jagged borders leave a little content in even a perfectly good gutter.
        """
        reference = float(np.percentile(profile, 75))
        return min(max(self._gutter_ratio, self._gutter_relative * reference), 0.2)

    @staticmethod
    def _segments(
        profile: NDArray[np.float64], threshold: float, min_gutter: int
    ) -> list[tuple[int, int]]:
        """Content segments of a profile, separated by gutter runs of at least min_gutter."""
        is_gutter = profile <= threshold
        segments: list[tuple[int, int]] = []
        start = 0
        i = 0
        n = len(profile)
        while i < n:
            if is_gutter[i]:
                j = i
                while j < n and is_gutter[j]:
                    j += 1
                if j - i >= min_gutter:
                    if i > start:
                        segments.append((start, i))
                    start = j
                i = j
            else:
                i += 1
        if start < n:
            segments.append((start, n))
        return segments
