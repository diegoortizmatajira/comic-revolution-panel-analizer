"""Page margin box: the content area of the page, used to normalize panels with no frame.

A framed panel has its own edges. Title lettering, a caption strip or a banner has none, so its
box is only as wide as its ink, and it ends a few pixels short of (or past) the panels around it.
The margin box is the bounding box of the framed panels, the part of the page the layout lives
in. Panels without a frame are stretched to its left and right edges and set against their
neighbours, so a title starts and ends where the panels below it do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.services.detection.frames import FrameLines

_Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 (end-exclusive)
MIN_FRAMED = 2  # framed panels needed before the page has a margin box
ASPECT = 2.5  # a banner is at least this many times wider than it is tall


def margin_box(rects: Sequence[_Rect], frames: FrameLines, min_sides: int = 2) -> _Rect | None:
    """Bounding box of the framed panels, if at least two panels are framed.

    One framed panel is no basis for a page layout, and where frame detection fails every panel
    looks unframed, so nothing is normalized.
    """
    framed = _framed(rects, frames, min_sides)
    if len(framed) < MIN_FRAMED or len(framed) == len(rects):
        return None
    return (
        min(r[0] for r in framed),
        min(r[1] for r in framed),
        max(r[2] for r in framed),
        max(r[3] for r in framed),
    )


def normalize_unframed(
    rects: Sequence[_Rect], frames: FrameLines, min_sides: int = 2, reach: float = 0.1
) -> list[_Rect]:
    """Stretch banner-like panels without a frame to the margin box and set them against others.

    Only a wide, short panel with no frame (title lettering, a banner) is normalized; a panel
    that merely lacks detected frame lines is left alone. Horizontally it takes the margin box's
    left and right edges when they are within ``reach`` of the page width. Vertically, one above
    the framed panels ends where the first framed panel below it begins, and one below them
    begins where the last framed panel above it ends.
    """
    box = margin_box(rects, frames, min_sides)
    if box is None:
        return list(rects)
    framed = _framed(rects, frames, min_sides)
    left, top, right, bottom = box
    slack = reach * frames.width
    result: list[_Rect] = []
    for rect in rects:
        x0, y0, x1, y1 = rect
        if rect in framed or (x1 - x0) < ASPECT * (y1 - y0):
            result.append(rect)
            continue
        x0 = left if abs(x0 - left) <= slack else x0
        x1 = right if abs(x1 - right) <= slack else x1
        if y1 <= top + 0.5 * (y1 - y0):  # lies (mostly) above the framed panels
            below = [f[1] for f in framed if f[1] >= y0]
            y1 = max(y0 + 1, min(below, default=y1))
        elif y0 >= bottom - 0.5 * (y1 - y0):  # lies (mostly) below them
            above = [f[3] for f in framed if f[3] <= y1]
            y0 = min(y1 - 1, max(above, default=y0))
        result.append((x0, y0, x1, y1))
    return result


def _framed(rects: Sequence[_Rect], frames: FrameLines, min_sides: int) -> list[_Rect]:
    return [r for r in rects if frames.sides((r[1], r[3], r[0], r[2])) >= min_sides]
