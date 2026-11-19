from pathlib import Path

import pytest
from app.domain.models import NormalizedBox, ReadingDirection
from app.evaluation.labels import (
    PageLabels,
    PanelLabel,
    discover,
    labels_path,
    read_labels,
    to_labels,
    write_labels,
)
from pydantic import ValidationError

from tests.conftest import make_page


def make_corpus(folder: Path, *, verified: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "page.png"
    image.write_bytes(make_page([(10, 10, 80, 80)], size=(100, 100)))
    write_labels(
        labels_path(image),
        PageLabels(
            verified=verified,
            tags=["clean"],
            reading_direction=ReadingDirection.RTL,
            panels=[PanelLabel(x=0.1, y=0.1, width=0.8, height=0.8)],
        ),
    )
    return image


def test_labels_path_sits_next_to_the_image() -> None:
    assert labels_path(Path("a/b/page.webp")) == Path("a/b/page.labels.json")


def test_round_trip(tmp_path: Path) -> None:
    image = make_corpus(tmp_path)
    labels = read_labels(labels_path(image))
    assert labels.version == 1
    assert labels.reading_direction is ReadingDirection.RTL
    assert labels.panels[0].to_box().width == 0.8


@pytest.mark.parametrize(
    "panel",
    [
        {"x": 0.5, "y": 0, "width": 0.6, "height": 0.5},  # runs past the right edge
        {"x": 0, "y": 0.9, "width": 0.5, "height": 0.2},  # runs past the bottom edge
        {"x": 0, "y": 0, "width": 0, "height": 0.5},  # empty
        {"x": -0.1, "y": 0, "width": 0.5, "height": 0.5},  # outside
        {"x": 0, "y": 0, "width": 0.5, "height": 0.5, "extra": 1},  # unknown key
    ],
)
def test_invalid_panels_are_rejected(panel: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        PageLabels.model_validate({"panels": [panel]})


def test_unknown_top_level_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PageLabels.model_validate({"panels": [], "colour": "red"})


def test_discover_loads_verified_pages(tmp_path: Path) -> None:
    make_corpus(tmp_path / "set")
    (page,) = discover([tmp_path / "set"])
    assert page.id == "set/page"
    assert page.direction is ReadingDirection.RTL
    assert page.tags == frozenset({"clean"})
    assert page.source == "file"
    assert (page.image.size.width, page.image.size.height) == (100, 100)
    assert len(page.panels) == 1


def test_discover_skips_unverified_drafts_unless_asked(tmp_path: Path) -> None:
    make_corpus(tmp_path / "set", verified=False)
    assert discover([tmp_path / "set"]) == []
    assert len(discover([tmp_path / "set"], include_unverified=True)) == 1


def test_discover_skips_images_without_labels_and_missing_folders(tmp_path: Path) -> None:
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "orphan.png").write_bytes(make_page([], size=(10, 10)))
    assert discover([folder, tmp_path / "missing"]) == []


def test_to_labels_converts_boxes() -> None:
    labels = to_labels([NormalizedBox(0.1, 0.2, 0.3, 0.4)], ReadingDirection.LTR, verified=False)
    assert labels.verified is False
    assert (labels.panels[0].x, labels.panels[0].height) == (0.1, 0.4)
