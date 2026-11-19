import pytest
from app.domain.models import BoundingBox, DetectedPanel, ImageSize, NormalizedBox


def test_bounding_box_derived_properties() -> None:
    box = BoundingBox(10, 20, 30, 40)
    assert (box.right, box.bottom, box.area) == (40, 60, 1200)


@pytest.mark.parametrize(("w", "h"), [(0, 5), (5, 0), (-1, 5)])
def test_bounding_box_rejects_non_positive_dimensions(w: int, h: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox(0, 0, w, h)


def test_image_size_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        ImageSize(0, 10)


@pytest.mark.parametrize("c", [-0.1, 1.1])
def test_detected_panel_rejects_out_of_range_confidence(c: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectedPanel(BoundingBox(0, 0, 1, 1), c)


def test_normalized_box() -> None:
    n = NormalizedBox.from_box(BoundingBox(100, 300, 200, 600), ImageSize(1000, 1200))
    assert (n.x, n.y, n.width, n.height) == (0.1, 0.25, 0.2, 0.5)
