"""CLI: ``uv run python -m app.evaluation`` (see docs/evaluation.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.domain.models import ReadingDirection
from app.evaluation.labels import LIMITS as IMAGE_LIMITS
from app.evaluation.labels import (
    LabeledPage,
    discover,
    image_files,
    labels_path,
    to_labels,
    write_labels,
)
from app.evaluation.report import compare, to_json, to_text
from app.evaluation.runner import evaluate
from app.evaluation.synthetic import synthetic_pages
from app.services.analyzer import PanelAnalyzer
from app.services.detection.registry import DETECTORS, build_detector
from app.services.image.decoder import decode_image
from app.services.ordering.xy_cut import XYCutOrdering

DEFAULT_CORPUS = [Path("tests/fixtures"), Path("samples")]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description="Score panel detectors against a labelled corpus.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        action="append",
        metavar="DIR",
        help="folder of images with <name>.labels.json files (repeatable; "
        "default: tests/fixtures and samples)",
    )
    p.add_argument("--no-synthetic", action="store_true", help="skip the generated pages")
    p.add_argument(
        "--detector",
        action="append",
        choices=sorted(DETECTORS),
        help="detector to score (repeatable; default: all)",
    )
    p.add_argument(
        "--include-unverified",
        action="store_true",
        help="also use draft labels that are not marked verified",
    )
    p.add_argument("--verbose", action="store_true", help="list every page")
    p.add_argument("--json", type=Path, metavar="PATH", help="write the full report as JSON")
    p.add_argument(
        "--write-baseline", type=Path, metavar="PATH", help="store this run as the baseline"
    )
    p.add_argument(
        "--baseline", type=Path, metavar="PATH", help="fail if metrics drop below this baseline"
    )
    p.add_argument(
        "--max-drop",
        type=float,
        default=0.02,
        help="allowed drop per metric versus the baseline (default 0.02)",
    )
    p.add_argument(
        "--propose",
        type=Path,
        metavar="PATH",
        help="write DRAFT labels (verified=false) for an image or folder, to be corrected by hand",
    )
    p.add_argument("--force", action="store_true", help="with --propose: overwrite existing labels")
    return p


def _propose(target: Path, detector_name: str, *, force: bool) -> int:
    images = image_files(target) if target.is_dir() else [target]
    analyzer = PanelAnalyzer(build_detector(detector_name), XYCutOrdering())
    for image in images:
        out = labels_path(image)
        if out.exists() and not force:
            print(f"skip {out} (exists; use --force)")
            continue
        decoded = decode_image(image.read_bytes(), IMAGE_LIMITS)
        result = analyzer.analyze(decoded, ReadingDirection.LTR)
        boxes = [p.normalized_bbox for p in result.panels]
        write_labels(out, to_labels(boxes, ReadingDirection.LTR, verified=False))
        print(f'wrote draft {out} with {len(boxes)} panels: review it, then set "verified": true')
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.propose is not None:
        return _propose(args.propose, (args.detector or ["projection"])[0], force=args.force)

    pages: list[LabeledPage] = discover(
        args.corpus or DEFAULT_CORPUS, include_unverified=args.include_unverified
    )
    if not args.no_synthetic:
        pages += synthetic_pages()
    if not pages:
        print("No labelled pages found.", file=sys.stderr)
        return 2

    reports = [evaluate(name, pages) for name in (args.detector or sorted(DETECTORS))]
    print(to_text(pages, reports, verbose=args.verbose))
    document = to_json(pages, reports)

    if args.json:
        args.json.write_text(json.dumps(document, indent=2) + "\n")
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(document, indent=2) + "\n")
        print(f"\nBaseline written to {args.write_baseline}")

    if args.baseline:
        problems = compare(document, json.loads(args.baseline.read_text()), args.max_drop)
        if problems:
            print("\nREGRESSIONS versus baseline:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"\nNo regressions versus {args.baseline}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
