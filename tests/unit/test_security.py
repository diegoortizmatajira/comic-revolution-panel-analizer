from app.services.security import SlidingWindowRateLimiter, StaticApiKeyStore


def test_api_key_store() -> None:
    store = StaticApiKeyStore(["secret-1", "secret-2"])
    assert store.authenticate("secret-1") is not None
    assert store.authenticate("secret-1") != store.authenticate("secret-2")
    assert store.authenticate("wrong") is None


def test_empty_store_rejects_everything() -> None:
    assert StaticApiKeyStore([]).authenticate("anything") is None


def test_rate_limiter_window() -> None:
    now = [0.0]
    limiter = SlidingWindowRateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.check("k") is None
    assert limiter.check("k") is None
    assert limiter.check("k") == 60
    assert limiter.check("other") is None  # independent per key
    now[0] = 61
    assert limiter.check("k") is None
