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


def border_paper(page: NDArray[np.uint8]) -> Mask:
    """Pixels reachable from the page border without crossing an edge: paper and gutters.

    Frame lines, lettering and art edges are barriers, so panel interiors are not reached,
    while the margin and the gutters between panels are. Unlike a global colour test this
    follows paper that is stained or unevenly toned.
    """
    height, width = page.shape[:2]
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 40, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    flooded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flooded[1:-1, 1:-1] = edges > 0  # barriers are pre-marked, so the fill cannot enter them
    canvas = np.zeros((height, width), dtype=np.uint8)
    step = 30
    seeds = [(x, y) for x in range(0, width, step) for y in (0, height - 1)]
    seeds += [(x, y) for y in range(0, height, step) for x in (0, width - 1)]
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    for x, y in seeds:
        if not flooded[y + 1, x + 1]:
            cv2.floodFill(canvas, flooded, (x, y), 255, 0, 0, flags)
    return np.asarray(flooded[1:-1, 1:-1] == 2, dtype=np.bool_)


def local_background(
    page: NDArray[np.uint8], paper: Mask, global_colour: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Paper colour per pixel: the border-connected paper smoothed over the page.

    Falls back to the single page-edge colour where no paper is nearby (the middle of a big
    panel).
    """
    height, width = page.shape[:2]
    scale = 4  # work at a quarter of the resolution: the paper tone varies slowly
    size = (max(1, width // scale), max(1, height // scale))
    small = cv2.resize(page, size, interpolation=cv2.INTER_AREA)
    weight = cv2.resize(paper.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    sigma = 0.05 * max(size)
    num = cv2.GaussianBlur(small.astype(np.float32) * weight[:, :, None], (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)[:, :, None]
    eps = 0.02  # pulls towards the global colour where there is little paper to learn from
    local = (num + eps * global_colour) / (den + eps)
    return np.asarray(
        cv2.resize(local, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.float32
    )


def content_mask(
    page: NDArray[np.uint8], tolerance: float = 40.0, border_fraction: float = 0.01
) -> Mask:
    """True where a pixel differs from the page colour by more than ``tolerance``."""
    background = estimate_background(page, border_fraction)
    distance = np.linalg.norm(page.astype(np.float32) - background, axis=2)
    return _clean(distance > tolerance)


def local_content_mask(
    page: NDArray[np.uint8], tolerance: float = 40.0, border_fraction: float = 0.01
) -> Mask:
    """Like :func:`content_mask` but against the paper colour measured around each pixel.

    For stained or unevenly toned paper, where the single page-edge colour matches only part of
    the page and most of it reads as content. It also leaves cleaner gutters between lettering,
    which splits titles, so it is a second attempt rather than the default.
    """
    colour = estimate_background(page, border_fraction)
    paper = border_paper(page)
    if not paper.any():
        return content_mask(page, tolerance, border_fraction)
    background = local_background(page, paper, colour)
    distance = np.linalg.norm(page.astype(np.float32) - background, axis=2)
    return _clean(distance > tolerance)


def _clean(mask: Mask) -> Mask:
    cleaned = cv2.medianBlur(mask.astype(np.uint8), 3)  # drop isolated noise pixels (JPEG/WebP)
    return np.asarray(cleaned, dtype=np.bool_)


def contrast_mask(page: NDArray[np.uint8], window: int = 31, threshold: float = 45.0) -> Mask:
    """True where a pixel stands out from its surroundings (outlines, lettering, art edges).

    Unlike :func:`content_mask` this does not compare with the colour at the page edge, so
    yellowed or unevenly toned paper does not read as content.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    local = cv2.medianBlur(gray, window)
    return np.asarray(cv2.absdiff(gray, local) > threshold, dtype=np.bool_)
