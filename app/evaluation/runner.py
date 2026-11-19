"""Run detectors over a labelled corpus and collect scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.evaluation.metrics import PageResult, Prediction, Summary, score_page, summarize
from app.services.analyzer import PanelAnalyzer
from app.services.detection.registry import build_detector
from app.services.ordering.xy_cut import XYCutOrdering

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.evaluation.labels import LabeledPage


@dataclass(frozen=True, slots=True)
class DetectorReport:
    detector: str
    version: str
    overall: Summary
    by_tag: dict[str, Summary]
    pages: tuple[PageResult, ...]


def evaluate(detector_name: str, pages: Sequence[LabeledPage]) -> DetectorReport:
    """Score one detector through the same analyzer and ordering the API uses."""
    detector = build_detector(detector_name)
    analyzer = PanelAnalyzer(detector, XYCutOrdering())
    results: list[PageResult] = []
    for page in pages:
        analysis = analyzer.analyze(page.image, page.direction)
        prediction = Prediction(
            boxes=tuple(p.normalized_bbox for p in analysis.panels),
            sequence_confidence=analysis.sequence_confidence,
            whole_page_fallback=analysis.whole_page_fallback,
        )
        results.append(score_page(page.id, page.tags, prediction, page.panels))

    tags = sorted({tag for page in pages for tag in page.tags})
    return DetectorReport(
        detector=detector_name,
        version=detector.version,
        overall=summarize(results),
        by_tag={tag: summarize(r for r in results if tag in r.tags) for tag in tags},
        pages=tuple(results),
    )
