"""Reading order via recursive XY-cut.

A group of panels is split along horizontal gutters (rows, top to bottom). If no
horizontal gutter exists, it is split along vertical gutters (columns, ordered by the
reading direction). Groups that can be cut neither way (overlapping or interlocking
panels) fall back to a (top, left) sort and lower the sequence confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.models import DetectedPanel, OrderingResult, ReadingDirection

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

AMBIGUITY_PENALTY = 0.5
# Chosen on the evaluation corpus: ML boxes order perfectly from about 8% up (docs/adr/0002).
DEFAULT_GAP_TOLERANCE = 0.08


class XYCutOrdering:
    """Recursive XY-cut ordering.

    ``gap_tolerance`` lets panels overlap slightly along an axis and still count as separated.
    It is a fraction of the group's extent on that axis. ML detectors return ragged boxes whose
    edges overlap by a few pixels; without tolerance no clean cut exists and the order is guessed.
    """

    name = "xycut"
    # 4: panels over each other never share a row; 3: uncuttable groups read as rows; 2: tolerance
    version = "4"

    def __init__(self, gap_tolerance: float = DEFAULT_GAP_TOLERANCE) -> None:
        self._tolerance = gap_tolerance

    def order(self, panels: Sequence[DetectedPanel], direction: ReadingDirection) -> OrderingResult:
        if not panels:
            return OrderingResult(panels=(), confidence=1.0)
        ordered, ambiguous = self._cut(tuple(panels), direction)
        confidence = 1.0 - AMBIGUITY_PENALTY * ambiguous / len(panels)
        return OrderingResult(panels=tuple(ordered), confidence=confidence)

    def _cut(
        self, group: tuple[DetectedPanel, ...], direction: ReadingDirection
    ) -> tuple[list[DetectedPanel], int]:
        """Return the ordered panels and how many were ordered ambiguously."""
        if len(group) == 1:
            return list(group), 0

        rows = _split(group, self._tolerance, lo=lambda p: p.bbox.y, hi=lambda p: p.bbox.bottom)
        if len(rows) > 1:
            return self._recurse(rows, direction)

        columns = _split(group, self._tolerance, lo=lambda p: p.bbox.x, hi=lambda p: p.bbox.right)
        if len(columns) > 1:
            if direction is ReadingDirection.RTL:
                columns.reverse()
            return self._recurse(columns, direction)

        return self._by_rows(group, direction), len(group)

    def _by_rows(
        self, group: tuple[DetectedPanel, ...], direction: ReadingDirection
    ) -> list[DetectedPanel]:
        """Order panels that cannot be cut: tops within the tolerance count as one row.

        Sorting by exact top edge would put a panel first just because its top sits a few
        pixels higher, even when it is the right-hand panel of the same row. Two panels that
        sit over each other (a title above the first panel) are never one row, however close
        their tops are: only panels side by side can share a row.
        """
        by_top = sorted(group, key=lambda p: (p.bbox.y, p.bbox.bottom))
        slack = self._tolerance * (max(p.bbox.bottom for p in group) - by_top[0].bbox.y)
        rows: list[list[DetectedPanel]] = [[by_top[0]]]
        for panel in by_top[1:]:
            same_row = panel.bbox.y - rows[-1][0].bbox.y <= slack and not any(
                _stacked(panel, other) for other in rows[-1]
            )
            if same_row:
                rows[-1].append(panel)
            else:
                rows.append([panel])
        rtl = direction is ReadingDirection.RTL
        return [
            p for row in rows for p in sorted(row, key=lambda p: -p.bbox.right if rtl else p.bbox.x)
        ]

    def _recurse(
        self, parts: list[tuple[DetectedPanel, ...]], direction: ReadingDirection
    ) -> tuple[list[DetectedPanel], int]:
        result: list[DetectedPanel] = []
        ambiguous = 0
        for part in parts:
            ordered, amb = self._cut(part, direction)
            result.extend(ordered)
            ambiguous += amb
        return result, ambiguous


def _stacked(a: DetectedPanel, b: DetectedPanel) -> bool:
    """True if the panels overlap horizontally by more than half of the narrower one."""
    shared = min(a.bbox.right, b.bbox.right) - max(a.bbox.x, b.bbox.x)
    return shared > 0.5 * min(a.bbox.width, b.bbox.width)


def _split(
    group: tuple[DetectedPanel, ...],
    tolerance: float,
    *,
    lo: Callable[[DetectedPanel], int],
    hi: Callable[[DetectedPanel], int],
) -> list[tuple[DetectedPanel, ...]]:
    """Split along gaps in the 1-D projection of the panels' extents."""
    ordered = sorted(group, key=lo)
    slack = tolerance * (max(hi(p) for p in group) - lo(ordered[0]))
    parts: list[list[DetectedPanel]] = [[ordered[0]]]
    reach = hi(ordered[0])
    for panel in ordered[1:]:
        if lo(panel) >= reach - slack:
            parts.append([panel])
        else:
            parts[-1].append(panel)
        reach = max(reach, hi(panel))
    return [tuple(p) for p in parts]
