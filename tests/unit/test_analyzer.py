from collections.abc import Sequence

import numpy as np
from app.domain.models import AnalysisResult, DetectedPanel, ImageSize, ReadingDirection
from app.services.analyzer import PanelAnalyzer
from app.services.image.decoder import DecodedImage
from app.services.ordering.xy_cut import XYCutOrdering
from numpy.typing import NDArray

from tests.conftest import panel


class FakeDetector:
    name = "fake"

    def __init__(self, panels: Sequence[DetectedPanel]) -> None:
        self._panels = panels

    def detect(self, page: NDArray[np.uint8]) -> Sequence[DetectedPanel]:
        return self._panels


def run(
    panels: Sequence[DetectedPanel], direction: ReadingDirection = ReadingDirection.LTR
) -> AnalysisResult:
    image = DecodedImage(np.zeros((1000, 500), dtype=np.uint8), ImageSize(500, 1000))
    return PanelAnalyzer(FakeDetector(panels), XYCutOrdering()).analyze(image, direction)


def test_numbers_panels_in_reading_order_with_normalized_boxes() -> None:
    result = run([panel(250, 0, 250, 500, 0.8), panel(0, 0, 250, 500, 0.9)])
    assert [(p.id, p.order, p.bbox.x) for p in result.panels] == [(1, 1, 0), (2, 2, 250)]
    assert result.panels[0].normalized_bbox.width == 0.5
    assert result.method == "fake"
    assert result.warnings == ()


def test_no_panels_warns() -> None:
    result = run([])
    assert result.panels == ()
    assert "No panels" in result.warnings[0]


def test_single_panel_warns() -> None:
    assert "splash" in run([panel(0, 0, 500, 1000)]).warnings[0]


def test_ambiguous_layout_warns() -> None:
    result = run([panel(0, 0, 60, 60), panel(40, 40, 60, 60)])
    assert any("ambiguous" in w for w in result.warnings)
