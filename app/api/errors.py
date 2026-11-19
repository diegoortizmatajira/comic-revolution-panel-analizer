"""Single place where exceptions become structured HTTP error responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError, RateLimitExceededError

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from starlette.responses import Response

logger = structlog.get_logger()


def _body(code: str, message: str, details: dict[str, Any], request: Request) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
        "request_id": getattr(request.state, "request_id", None),
    }


async def _app_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, AppError)  # noqa: S101
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitExceededError) and "retry_after" in exc.details:
        headers["Retry-After"] = str(int(exc.details["retry_after"]) + 1)
    if exc.status_code >= 500:
        logger.error("request_failed", code=exc.code, message=exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=_body(exc.code, exc.message, exc.details, request),
        headers=headers,
    )


async def _validation_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    details = {"errors": jsonable_encoder(exc.errors(), exclude={"input", "ctx"})}
    return JSONResponse(
        status_code=422,
        content=_body("validation_error", "Request validation failed.", details, request),
    )


async def _unhandled(request: Request, exc: Exception) -> Response:
    logger.exception("unhandled_exception", error=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=_body("internal_error", "Internal server error.", {}, request),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
