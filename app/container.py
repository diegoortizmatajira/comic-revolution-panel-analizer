"""Composition root: the only place concrete adapters are wired together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.analyzer import PanelAnalyzer
from app.services.detection.classical import ClassicalPanelDetector
from app.services.detection.projection import ProjectionPanelDetector
from app.services.image.decoder import ImageLimits
from app.services.image.sources import UrlImageFetcher
from app.services.ordering.xy_cut import XYCutOrdering
from app.services.security import SlidingWindowRateLimiter, StaticApiKeyStore

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.domain.ports import ApiKeyStore, ImageFetcher, PanelDetector, RateLimiter


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    analyzer: PanelAnalyzer
    fetcher: ImageFetcher
    limits: ImageLimits
    api_keys: ApiKeyStore
    rate_limiter: RateLimiter


def _build_detector(settings: Settings) -> PanelDetector:
    if settings.detector == "classical":
        return ClassicalPanelDetector()
    return ProjectionPanelDetector()


def build_container(settings: Settings) -> Container:
    return Container(
        settings=settings,
        analyzer=PanelAnalyzer(_build_detector(settings), XYCutOrdering()),
        fetcher=UrlImageFetcher(
            max_bytes=settings.max_image_bytes,
            timeout_seconds=settings.url_fetch_timeout_seconds,
            max_redirects=settings.url_max_redirects,
            allow_private=settings.allow_private_urls,
        ),
        limits=ImageLimits(
            max_bytes=settings.max_image_bytes, max_pixels=settings.max_image_pixels
        ),
        api_keys=StaticApiKeyStore(settings.parsed_api_keys()),
        rate_limiter=SlidingWindowRateLimiter(settings.rate_limit_per_minute),
    )
