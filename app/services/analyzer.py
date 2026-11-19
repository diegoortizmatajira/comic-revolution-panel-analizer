"""Application service: detect panels, order them, score and package the result."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.models import AnalysisResult, BoundingBox, NormalizedBox, OrderedPanel

if TYPE_CHECKING:
    from app.domain.models import ReadingDirection
    from app.domain.ports import PanelDetector, PanelRefiner, ReadingOrderStrategy
    from app.services.image.decoder import DecodedImage

LOW_SEQUENCE_CONFIDENCE = 0.6
# Confidence given to the whole-page stand-in returned when nothing is detected.
FALLBACK_CONFIDENCE = 0.1


class PanelAnalyzer:
    def __init__(
        self,
        detector: PanelDetector,
        ordering: ReadingOrderStrategy,
        refiner: PanelRefiner | None = None,
    ) -> None:
        self._detector = detector
        self._ordering = ordering
        self._refiner = refiner

    def analyze(self, image: DecodedImage, direction: ReadingDirection) -> AnalysisResult:
        detection = self._detector.detect(image.pixels)
        found = detection.panels
        if self._refiner is not None:
            found = self._refiner.refine(image.pixels, found)
        ordering = self._ordering.order(found, direction)

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
        sequence_confidence = ordering.confidence
        fallback = not panels
        if fallback:
            # Clients can always render something: treat the whole page as one panel.
            whole_page = BoundingBox(x=0, y=0, width=image.size.width, height=image.size.height)
            panels = (
                OrderedPanel(
                    id=1,
                    order=1,
                    bbox=whole_page,
                    normalized_bbox=NormalizedBox.from_box(whole_page, image.size),
                    confidence=FALLBACK_CONFIDENCE,
                ),
            )
            sequence_confidence = FALLBACK_CONFIDENCE
            warnings.append(
                "No panels were detected; returning the whole page as one low-confidence panel."
            )
        elif len(panels) == 1:
            warnings.append("Only one panel was detected; the page may be a splash page.")
        if not fallback and ordering.confidence < LOW_SEQUENCE_CONFIDENCE:
            warnings.append("Panel layout is ambiguous; the reading order may be unreliable.")

        return AnalysisResult(
            image_size=image.size,
            reading_direction=direction,
            panels=panels,
            sequence_confidence=sequence_confidence,
            method=detection.method,
            version=self._version(),
            whole_page_fallback=fallback,
            warnings=tuple(warnings),
        )

    def _version(self) -> str:
        parts = [f"{self._detector.name}@{self._detector.version}"]
        if self._refiner is not None:
            parts.append(f"{self._refiner.name}@{self._refiner.version}")
        parts.append(f"{self._ordering.name}@{self._ordering.version}")
        return "+".join(parts)
