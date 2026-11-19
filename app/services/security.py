"""API-key authentication and in-memory rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class StaticApiKeyStore:
    def __init__(self, keys: Iterable[str]) -> None:
        self._digests = frozenset(_digest(k) for k in keys)

    def authenticate(self, presented_key: str) -> str | None:
        presented = _digest(presented_key)
        matched = False
        for known in self._digests:  # constant-time comparison per entry
            matched |= hmac.compare_digest(known, presented)
        return presented[:12] if matched else None


class SlidingWindowRateLimiter:
    """Per-key sliding window; single-process only (swap for a shared store to scale out)."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key_id: str) -> float | None:
        now = self._clock()
        hits = self._hits[key_id]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return self._window - (now - hits[0])
        hits.append(now)
        return None
