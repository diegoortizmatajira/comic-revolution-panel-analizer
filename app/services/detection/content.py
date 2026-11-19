"""Page-level ink detection shared by the projection detector and the edge refiner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Mask = NDArray[np.bool_]


def estimate_background(
    page: NDArray[np.uint8], border_fraction: float = 0.01
) -> NDArray[np.float32]:
    """Page colour, sampled from a thin strip around the page edge."""
    height, width = page.shape[:2]
    t = max(1, int(min(height, width) * border_fraction))
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


def content_mask(
    page: NDArray[np.uint8], tolerance: float = 40.0, border_fraction: float = 0.01
) -> Mask:
    """True where a pixel differs from the page colour by more than ``tolerance``."""
    background = estimate_background(page, border_fraction)
    distance = np.linalg.norm(page.astype(np.float32) - background, axis=2)
    mask = (distance > tolerance).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)  # drop isolated noise pixels (JPEG/WebP artifacts)
    return np.asarray(mask, dtype=np.bool_)


def contrast_mask(page: NDArray[np.uint8], window: int = 31, threshold: float = 45.0) -> Mask:
    """True where a pixel stands out from its surroundings (outlines, lettering, art edges).

    Unlike :func:`content_mask` this does not compare with the colour at the page edge, so
    yellowed or unevenly toned paper does not read as content.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    local = cv2.medianBlur(gray, window)
    return np.asarray(cv2.absdiff(gray, local) > threshold, dtype=np.bool_)
