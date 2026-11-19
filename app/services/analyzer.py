"""Application service: detect panels, order them, score and package the result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.models import AnalysisResult, NormalizedBox, OrderedPanel

if TYPE_CHECKING:
    from app.domain.models import ReadingDirection
    from app.domain.ports import PanelDetector, ReadingOrderStrategy
    from app.services.image.decoder import DecodedImage

LOW_SEQUENCE_CONFIDENCE = 0.6


class PanelAnalyzer:
    def __init__(self, detector: PanelDetector, ordering: ReadingOrderStrategy) -> None:
        self._detector = detector
        self._ordering = ordering

    def analyze(self, image: DecodedImage, direction: ReadingDirection) -> AnalysisResult:
        detected = self._detector.detect(image.gray)
        ordering = self._ordering.order(detected, direction)

        panels = tuple(
            OrderedPanel(
                id=index,
                order=index,
                bbox=panel.bbox,
                normalized_bbox=NormalizedBox.from_box(panel.bbox, image.size),
                confidence=panel.confidence,
            )
            for index, panel in enumerate(ordering.panels, start=1)
        )

        warnings: list[str] = []
        if not panels:
            warnings.append("No panels were detected.")
        elif len(panels) == 1:
            warnings.append("Only one panel was detected; the page may be a splash page.")
        if panels and ordering.confidence < LOW_SEQUENCE_CONFIDENCE:
            warnings.append("Panel layout is ambiguous; the reading order may be unreliable.")

        return AnalysisResult(
            image_size=image.size,
            reading_direction=direction,
            panels=panels,
            sequence_confidence=ordering.confidence,
            method=self._detector.name,
            warnings=tuple(warnings),
        )
