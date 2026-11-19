"""Gutter-based panel detector using projection profiles (recursive XY-cut on pixels).

1. Estimate the page background colour from a thin strip around the page edge.
2. Mark every pixel whose colour differs from it as "content".
3. Recursively split a region wherever a run of rows (or columns) contains almost no
   content: that run is a gutter. Rows are tried first, then columns, then each piece
   is trimmed to its content and split again. Regions that cannot be split are panels.

Unlike contour detection this does not need panels to be disconnected blobs, so panels
whose borders or art nearly touch are still separated as long as a gutter runs between
them. Known limitations: panels with no gutter at all stay merged; borderless panels
whose artwork contains background-coloured bands can be split by mistake; a full-bleed
splash page (no margin to sample) yields no panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import BoundingBox, DetectedPanel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    Mask = NDArray[np.bool_]

_Region = tuple[int, int, int, int]  # y0, y1, x0, x1 (end-exclusive)


class ProjectionPanelDetector:
    name = "projection"

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
    ) -> None:
        self._tolerance = color_tolerance
        self._gutter_ratio = gutter_content_ratio
        self._gutter_relative = gutter_relative_threshold
        self._min_gutter_ratio = min_gutter_ratio
        self._min_area_ratio = min_area_ratio
        self._border_fraction = border_fraction
        self._max_depth = max_depth

    def detect(self, page: NDArray[np.uint8]) -> Sequence[DetectedPanel]:
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
        return panels

    def _content_mask(self, page: NDArray[np.uint8]) -> Mask:
        background = self._estimate_background(page)
        distance = np.linalg.norm(page.astype(np.float32) - background, axis=2)
        mask = (distance > self._tolerance).astype(np.uint8)
        mask = cv2.medianBlur(mask, 3)  # drop isolated noise pixels (JPEG/WebP artifacts)
        return np.asarray(mask, dtype=np.bool_)

    def _estimate_background(self, page: NDArray[np.uint8]) -> NDArray[np.float32]:
        height, width = page.shape[:2]
        t = max(1, int(min(height, width) * self._border_fraction))
        border = np.concatenate(
            [
                page[:t].reshape(-1, 3),
                page[-t:].reshape(-1, 3),
                page[:, :t].reshape(-1, 3),
                page[:, -t:].reshape(-1, 3),
            ]
        )
        # Dominant colour, not the median: art that bleeds off the page edge must not win.
        bins = (
            (border[:, 0].astype(np.int32) >> 4) * 256
            + (border[:, 1].astype(np.int32) >> 4) * 16
            + (border[:, 2].astype(np.int32) >> 4)
        )
        dominant = int(np.bincount(bins).argmax())
        return np.asarray(np.median(border[bins == dominant], axis=0), dtype=np.float32)

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
        out.append((y0, y1, x0, x1))

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
