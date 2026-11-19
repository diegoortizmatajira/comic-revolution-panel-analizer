"""FastAPI dependencies: container access, authentication, rate limiting."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from app.container import Container
from app.core.errors import RateLimitExceededError, UnauthorizedError


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def authenticated_key_id(
    container: ContainerDep,
    x_api_key: Annotated[str | None, Header(description="API key.")] = None,
) -> str:
    key_id = container.api_keys.authenticate(x_api_key) if x_api_key else None
    if key_id is None:
        msg = "Missing or invalid API key."
        raise UnauthorizedError(msg)
    retry_after = container.rate_limiter.check(key_id)
    if retry_after is not None:
        msg = "Rate limit exceeded."
        raise RateLimitExceededError(msg, {"retry_after": retry_after})
    return key_id


AuthDep = Annotated[str, Depends(authenticated_key_id)]
