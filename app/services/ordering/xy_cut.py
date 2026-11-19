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


class XYCutOrdering:
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

        rows = _split(group, lo=lambda p: p.bbox.y, hi=lambda p: p.bbox.bottom)
        if len(rows) > 1:
            return self._recurse(rows, direction)

        columns = _split(group, lo=lambda p: p.bbox.x, hi=lambda p: p.bbox.right)
        if len(columns) > 1:
            if direction is ReadingDirection.RTL:
                columns.reverse()
            return self._recurse(columns, direction)

        fallback = sorted(group, key=lambda p: (p.bbox.y, p.bbox.x))
        if direction is ReadingDirection.RTL:
            fallback = sorted(group, key=lambda p: (p.bbox.y, -p.bbox.right))
        return fallback, len(group)

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


def _split(
    group: tuple[DetectedPanel, ...],
    *,
    lo: Callable[[DetectedPanel], int],
    hi: Callable[[DetectedPanel], int],
) -> list[tuple[DetectedPanel, ...]]:
    """Split along gaps in the 1-D projection of the panels' extents."""
    ordered = sorted(group, key=lo)
    parts: list[list[DetectedPanel]] = [[ordered[0]]]
    reach = hi(ordered[0])
    for panel in ordered[1:]:
        if lo(panel) >= reach:
            parts.append([panel])
        else:
            parts[-1].append(panel)
        reach = max(reach, hi(panel))
    return [tuple(p) for p in parts]
