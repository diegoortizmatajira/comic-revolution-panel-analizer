"""Composition root: the only place concrete adapters are wired together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.analyzer import PanelAnalyzer
from app.services.detection.refine import EdgeSnapRefiner
from app.services.detection.registry import MlConfig, build_detector
from app.services.image.decoder import ImageLimits
from app.services.image.sources import UrlImageFetcher
from app.services.ordering.xy_cut import XYCutOrdering
from app.services.security import SlidingWindowRateLimiter, StaticApiKeyStore

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.domain.ports import ApiKeyStore, ImageFetcher, RateLimiter


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    analyzer: PanelAnalyzer
    fetcher: ImageFetcher
    limits: ImageLimits
    api_keys: ApiKeyStore
    rate_limiter: RateLimiter


def ml_config(settings: Settings) -> MlConfig | None:
    if settings.ml_model is None:
        return None
    return MlConfig(
        model=settings.ml_model,
        input_size=settings.ml_input_size,
        confidence=settings.ml_confidence,
        nms_iou=settings.ml_nms_iou,
        output_format=settings.ml_output_format,
        panel_class=settings.ml_panel_class,
    )


def build_container(settings: Settings) -> Container:
    return Container(
        settings=settings,
        analyzer=PanelAnalyzer(
            build_detector(settings.detector, ml_config(settings)),
            XYCutOrdering(),
            (
                EdgeSnapRefiner(max_expand_ratio=settings.refine_max_expand)
                if settings.refine_edges
                else None
            ),
        ),
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
