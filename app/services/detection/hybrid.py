"""Hybrid detector: a cheap primary detector, with an ML detector as a fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from app.domain.models import DetectedPanel, Detection
    from app.domain.ports import PanelDetector

VERSION = "2"  # bump when the policy or the selection rule changes (2: a title panel is exempt)


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    """When is the primary result not trusted, so the ML detector is asked as well?

    Two cheap signals, both visible in the primary's own output (see docs/adr/0002):

    * very few panels: a page with fewer than ``min_panels`` panels is usually a page where
      panels were merged into one (comic pages with one or two panels are rare);
    * a very thin panel: a box whose long side is at least ``max_aspect`` times its short side
      is nearly always a caption strip counted as a panel. The one exception is a thin panel
      that lies wholly above every other panel: that is the title banner the primary detector
      returns on purpose as a first panel, and the ML model would only drop it.

    They are deliberately conservative: pages the primary gets right must not be sent to the
    fallback, so hybrid never does worse than the primary on them.
    """

    min_panels: int = 3
    max_aspect: float = 5.0

    def distrusts(self, primary: Detection) -> bool:
        panels = primary.panels
        if len(panels) < self.min_panels:
            return True
        return any(self._is_thin(p) and not _is_title(p, panels) for p in panels)

    def _is_thin(self, panel: DetectedPanel) -> bool:
        box = panel.bbox
        return max(box.width, box.height) >= self.max_aspect * min(box.width, box.height)


def _is_title(panel: DetectedPanel, panels: tuple[DetectedPanel, ...]) -> bool:
    """A panel wholly above all the others: the title banner, not a caption strip between them."""
    others = [p for p in panels if p is not panel]
    return bool(others) and all(panel.bbox.bottom <= p.bbox.y for p in others)


class HybridPanelDetector:
    name = "hybrid"

    def __init__(
        self,
        primary: PanelDetector,
        fallback: PanelDetector,
        policy: ReliabilityPolicy | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._policy = policy or ReliabilityPolicy()
        # Results depend on both detectors (and their weights) and on the policy.
        self.version = (
            f"{VERSION}[{primary.name}@{primary.version}|{fallback.name}@{fallback.version}]"
        )

    def detect(self, page: NDArray[np.uint8]) -> Detection:
        first = self._primary.detect(page)
        if not self._policy.distrusts(first):
            return first
        second = self._fallback.detect(page)
        # Keep the primary answer unless the fallback actually found something.
        return second if second.panels else first
