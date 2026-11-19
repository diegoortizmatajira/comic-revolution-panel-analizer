import numpy as np
import pytest
from app.domain.models import DetectedPanel, Detection
from app.services.detection.hybrid import HybridPanelDetector, ReliabilityPolicy

from tests.conftest import panel

PAGE = np.zeros((10, 10, 3), dtype=np.uint8)


class StubDetector:
    def __init__(self, name: str, version: str, panels: list[DetectedPanel]) -> None:
        self.name = name
        self.version = version
        self._panels = panels
        self.calls = 0

    def detect(self, page: np.ndarray) -> Detection:
        self.calls += 1
        return Detection(tuple(self._panels), self.name)


def grid(n: int) -> list[DetectedPanel]:
    return [panel(i * 100, 0, 90, 90) for i in range(n)]


def test_a_plausible_primary_result_is_used_without_asking_the_fallback() -> None:
    primary, fallback = StubDetector("projection", "1", grid(4)), StubDetector("ml", "9", grid(6))
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "projection"
    assert len(result.panels) == 4
    assert fallback.calls == 0  # the model is never run for pages the primary handles


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_panels_asks_the_fallback(count: int) -> None:
    primary, fallback = (
        StubDetector("projection", "1", grid(count)),
        StubDetector("ml", "9", grid(5)),
    )
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "ml"  # the method reflects who actually answered this page
    assert len(result.panels) == 5


def test_a_very_thin_panel_asks_the_fallback() -> None:
    strip = [panel(0, 0, 500, 100), panel(0, 200, 90, 90), panel(100, 200, 90, 90)]  # 5:1 banner
    primary, fallback = StubDetector("projection", "1", strip), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "ml"


def test_just_below_the_thin_threshold_is_trusted() -> None:
    almost = [panel(0, 0, 499, 100), panel(0, 200, 90, 90), panel(100, 200, 90, 90)]
    primary, fallback = StubDetector("projection", "1", almost), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "projection"


def test_thin_in_either_direction_counts() -> None:
    tall = [panel(0, 0, 100, 500), panel(200, 0, 90, 90), panel(300, 0, 90, 90)]
    primary, fallback = StubDetector("projection", "1", tall), StubDetector("ml", "9", grid(4))
    assert HybridPanelDetector(primary, fallback).detect(PAGE).method == "ml"


def test_an_empty_fallback_answer_keeps_the_primary_one() -> None:
    primary, fallback = StubDetector("projection", "1", grid(1)), StubDetector("ml", "9", [])
    result = HybridPanelDetector(primary, fallback).detect(PAGE)
    assert result.method == "projection"
    assert len(result.panels) == 1
    assert fallback.calls == 1


def test_version_covers_both_detectors_so_cached_results_invalidate() -> None:
    a = HybridPanelDetector(StubDetector("projection", "1", []), StubDetector("ml", "1:abc", []))
    b = HybridPanelDetector(StubDetector("projection", "1", []), StubDetector("ml", "1:def", []))
    assert a.name == "hybrid"
    assert a.version == "1[projection@1|ml@1:abc]"
    assert a.version != b.version  # new weights -> new version


def test_the_policy_is_configurable() -> None:
    lenient = ReliabilityPolicy(min_panels=1, max_aspect=100.0)
    primary, fallback = StubDetector("projection", "1", grid(1)), StubDetector("ml", "9", grid(5))
    assert HybridPanelDetector(primary, fallback, lenient).detect(PAGE).method == "projection"
