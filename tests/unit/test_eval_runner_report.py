import json
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
from app.core.config import Settings
from app.domain.models import BoundingBox, DetectedPanel
from app.evaluation import __main__ as cli
from app.evaluation.labels import LabeledPage
from app.evaluation.report import compare, to_json, to_text
from app.evaluation.runner import evaluate
from app.evaluation.synthetic import synthetic_pages
from app.services.detection.registry import DETECTORS, build_detector

from tests.conftest import make_page
from tests.unit.test_eval_labels import make_corpus


class OracleDetector:
    """Returns the page's ground truth in pixels: a perfect detector for one page."""

    name = "oracle"
    version = "test"

    def __init__(self, boxes: list[BoundingBox]) -> None:
        self._boxes = boxes

    def detect(self, page: np.ndarray) -> list[DetectedPanel]:
        return [DetectedPanel(b, 1.0) for b in self._boxes]


def oracle_for(page_index: int, monkeypatch: pytest.MonkeyPatch) -> list[LabeledPage]:
    pages = synthetic_pages()
    page = pages[page_index]
    width, height = page.image.size.width, page.image.size.height
    boxes = [
        BoundingBox(
            round(b.x * width),
            round(b.y * height),
            round(b.width * width),
            round(b.height * height),
        )
        for b in page.panels
    ]
    monkeypatch.setitem(DETECTORS, "oracle", lambda: OracleDetector(boxes))
    return [page]


def test_synthetic_ground_truth_is_well_formed() -> None:
    pages = synthetic_pages()
    assert len(pages) >= 10
    assert len({p.id for p in pages}) == len(pages)
    for p in pages:
        assert "synthetic" in p.tags
        assert p.panels
        for b in p.panels:
            assert 0 <= b.x <= 1
            assert 0 <= b.y <= 1
            assert b.x + b.width <= 1 + 1e-9
            assert b.y + b.height <= 1 + 1e-9


def test_synthetic_corpus_covers_the_hard_cases() -> None:
    tags = {t for p in synthetic_pages() for t in p.tags}
    assert {
        "splash",
        "borderless",
        "rtl",
        "spread",
        "diagonal",
        "dark",
        "bleed",
        "no-gutter",
    } <= tags


@pytest.mark.parametrize("index", range(11))
def test_runner_scores_a_perfect_detector_as_perfect(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = oracle_for(index, monkeypatch)
    expected_order_ok = "spread" not in pages[0].tags  # XY-cut still mis-orders spreads (see #5)
    report = evaluate("oracle", pages)
    assert report.overall.f1_50 == 1.0
    assert report.overall.f1_75 == 1.0
    if expected_order_ok:
        assert report.overall.page_accuracy == 1.0


def test_runner_reports_by_tag_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = oracle_for(0, monkeypatch)
    report = evaluate("oracle", pages)
    assert report.detector == "oracle"
    assert report.version == "test"
    assert set(report.by_tag) == {"synthetic", "clean"}
    assert report.by_tag["clean"].pages == 1


def test_registry_builds_known_detectors_and_rejects_unknown() -> None:
    assert build_detector("projection").name == "projection"
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("nope")


def test_registry_matches_the_settings_choices() -> None:
    assert set(get_args(Settings.model_fields["detector"].annotation)) == set(DETECTORS)


def test_text_report_names_detectors_and_tags() -> None:
    pages = synthetic_pages()[:3]
    reports = [evaluate("projection", pages), evaluate("classical", pages)]
    text = to_text(pages, reports, verbose=True)
    for needle in (
        "projection",
        "classical",
        "page_acc",
        "conf_auc",
        "by tag",
        "synthetic/grid_white",
    ):
        assert needle in text


def document() -> dict[str, Any]:
    pages = synthetic_pages()[:4]
    return to_json(pages, [evaluate("projection", pages)])


def test_compare_passes_when_unchanged_or_improved() -> None:
    doc = document()
    assert compare(doc, doc, 0.02) == []


def test_compare_flags_a_regression_beyond_the_allowance() -> None:
    now, before = document(), document()
    before["detectors"]["projection"]["overall"]["f1_50"] = 1.2  # baseline was better
    problems = compare(now, before, 0.02)
    assert len(problems) == 1
    assert "projection: f1_50 dropped" in problems[0]
    assert compare(now, before, 0.5) == []  # within a generous allowance


def test_compare_refuses_a_different_corpus() -> None:
    now, before = document(), document()
    before["corpus"]["pages"] = ["something/else"]
    (problem,) = compare(now, before, 0.02)
    assert "corpus differs" in problem


def test_compare_ignores_detectors_missing_from_the_baseline() -> None:
    now, before = document(), document()
    before["detectors"] = {}
    assert compare(now, before, 0.02) == []


def test_cli_runs_a_corpus_and_writes_json_and_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path / "set")
    out, baseline = tmp_path / "out.json", tmp_path / "baseline.json"
    argv = ["--corpus", str(tmp_path / "set"), "--no-synthetic", "--json", str(out)]
    assert cli.main([*argv, "--write-baseline", str(baseline)]) == 0
    data = json.loads(out.read_text())
    assert set(data["detectors"]) == set(DETECTORS)
    assert data["corpus"]["pages"] == ["set/page"]
    assert cli.main([*argv, "--baseline", str(baseline)]) == 0
    assert "No regressions" in capsys.readouterr().out


def test_cli_exits_non_zero_on_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_corpus(tmp_path / "set")
    baseline = tmp_path / "baseline.json"
    argv = ["--corpus", str(tmp_path / "set"), "--no-synthetic"]
    cli.main([*argv, "--write-baseline", str(baseline)])
    data = json.loads(baseline.read_text())
    data["detectors"]["projection"]["overall"]["page_accuracy"] = 2.0
    baseline.write_text(json.dumps(data))
    assert cli.main([*argv, "--baseline", str(baseline)]) == 1
    assert "REGRESSIONS" in capsys.readouterr().err


def test_cli_without_any_labelled_page_is_an_error(tmp_path: Path) -> None:
    assert cli.main(["--corpus", str(tmp_path), "--no-synthetic"]) == 2


def test_cli_detector_selection(tmp_path: Path) -> None:
    make_corpus(tmp_path / "set")
    out = tmp_path / "o.json"
    cli.main(
        [
            "--corpus",
            str(tmp_path / "set"),
            "--no-synthetic",
            "--detector",
            "classical",
            "--json",
            str(out),
        ]
    )
    assert set(json.loads(out.read_text())["detectors"]) == {"classical"}


def test_cli_propose_writes_unverified_drafts_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "pages"
    folder.mkdir()
    (folder / "a.png").write_bytes(make_page([(40, 40, 300, 400)], size=(400, 500)))
    assert cli.main(["--propose", str(folder)]) == 0
    draft = json.loads((folder / "a.labels.json").read_text())
    assert draft["verified"] is False
    assert len(draft["panels"]) == 1
    # Drafts are ignored by the harness until a person verifies them.
    assert cli.main(["--corpus", str(folder), "--no-synthetic"]) == 2
    (folder / "a.labels.json").write_text('{"keep": "me"}')
    cli.main(["--propose", str(folder)])
    assert "exists" in capsys.readouterr().out
    assert (folder / "a.labels.json").read_text() == '{"keep": "me"}'
