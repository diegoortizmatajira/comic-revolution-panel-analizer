"""Classical computer-vision panel detector (OpenCV).

Treats the page border colour as the gutter/background, marks everything that differs
from it as content, and takes the outer contours of the content blobs as panels.
Known limitation: panels that touch each other with no gutter merge into one blob.
A pluggable ML detector is intended to handle those cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.domain.models import BoundingBox, DetectedPanel, Detection

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClassicalPanelDetector:
    name = "classical"
    version = "1"

    def __init__(
        self,
        *,
        background_tolerance: int = 30,
        min_area_ratio: float = 0.02,
        border_fraction: float = 0.01,
    ) -> None:
        self._tolerance = background_tolerance
        self._min_area_ratio = min_area_ratio
        self._border_fraction = border_fraction

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        gray = np.asarray(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        height, width = gray.shape
        background = self._estimate_background(gray)
        diff = np.abs(gray.astype(np.int16) - background)
        mask = (diff > self._tolerance).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self._min_area_ratio * width * height
        panels: list[DetectedPanel] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_area:
                continue
            # A rectangular panel fills most of its bounding box.
            fill = float(cv2.contourArea(contour)) / (w * h)
            panels.append(
                DetectedPanel(
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    confidence=min(1.0, max(0.0, fill)),
                )
            )
        return Detection(panels=tuple(panels), method=self.name)

    def _estimate_background(self, page: NDArray[np.uint8]) -> int:
        height, width = page.shape
        t = max(1, int(min(height, width) * self._border_fraction))
        border = np.concatenate(
            [page[:t].ravel(), page[-t:].ravel(), page[:, :t].ravel(), page[:, -t:].ravel()]
        )
        return int(np.median(border))
