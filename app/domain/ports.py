"""Ports (interfaces) the domain depends on. Adapters implement them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from app.domain.models import DetectedPanel, OrderingResult, ReadingDirection


class PanelDetector(Protocol):
    """Finds panels in a decoded page (RGB uint8 array, shape HxWx3)."""

    @property
    def name(self) -> str: ...

    def detect(self, page: NDArray[np.uint8]) -> Sequence[DetectedPanel]: ...


class ReadingOrderStrategy(Protocol):
    def order(
        self, panels: Sequence[DetectedPanel], direction: ReadingDirection
    ) -> OrderingResult: ...


class ImageFetcher(Protocol):
    """Retrieves raw image bytes from a remote location."""

    async def fetch(self, url: str) -> bytes: ...


class HostResolver(Protocol):
    """Resolves a hostname to its IP addresses (injectable for SSRF tests)."""

    async def resolve(self, host: str, port: int) -> list[str]: ...


class ApiKeyStore(Protocol):
    def authenticate(self, presented_key: str) -> str | None:
        """Return a stable, non-secret key identifier, or None if invalid."""
        ...


class RateLimiter(Protocol):
    def check(self, key_id: str) -> float | None:
        """Consume one request. Return None if allowed, else seconds to wait."""
        ...
