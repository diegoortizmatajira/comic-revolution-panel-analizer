"""Named detectors, shared by the API container and the evaluation harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.detection.classical import ClassicalPanelDetector
from app.services.detection.projection import ProjectionPanelDetector

if TYPE_CHECKING:
    from collections.abc import Callable

    from app.domain.ports import PanelDetector

DETECTORS: dict[str, Callable[[], PanelDetector]] = {
    "projection": ProjectionPanelDetector,
    "classical": ClassicalPanelDetector,
}


def build_detector(name: str) -> PanelDetector:
    try:
        return DETECTORS[name]()
    except KeyError:
        known = ", ".join(sorted(DETECTORS))
        msg = f"Unknown detector {name!r}; known detectors: {known}"
        raise ValueError(msg) from None
