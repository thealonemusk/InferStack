"""OpenAI-compatible error responses.

The gateway's whole value is that existing OpenAI client code works against it
by changing one URL. That has to include the *failure* path: SDKs branch on the
error envelope, and a client that understands ``429 rate_limit_exceeded`` from
OpenAI should understand it from here too.

So errors are emitted in OpenAI's shape rather than FastAPI's default
``{"detail": ...}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

# OpenAI's error `type` values that clients actually branch on.
TYPE_INVALID_REQUEST = "invalid_request_error"
TYPE_AUTHENTICATION = "authentication_error"
TYPE_RATE_LIMIT = "rate_limit_error"
TYPE_SERVER = "server_error"
TYPE_UPSTREAM = "upstream_error"


class GatewayError(Exception):
    """An error that should reach the client as an OpenAI-shaped envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = TYPE_SERVER,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers or {}


def error_body(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    """The envelope itself, matching OpenAI's schema."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def error_response(
    status_code: int,
    message: str,
    error_type: str = TYPE_SERVER,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(message, error_type, code),
        headers=headers or {},
    )


async def gateway_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a :class:`GatewayError` as an OpenAI-shaped response."""
    assert isinstance(exc, GatewayError)  # noqa: S101 - registered for this type only
    return error_response(
        exc.status_code, exc.message, exc.error_type, exc.code, headers=exc.headers
    )


# --- the errors this gateway actually raises ------------------------------


def unauthorized(message: str = "Missing or invalid API key.") -> GatewayError:
    # 401 with a WWW-Authenticate header is what an HTTP client expects, and
    # what makes the failure self-describing.
    return GatewayError(
        401,
        message,
        TYPE_AUTHENTICATION,
        "invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def overloaded(retry_after_s: float, message: str | None = None) -> GatewayError:
    """Shed load rather than queue it.

    Phase 5 theory, applied: accepting a request that cannot meet its SLO
    degrades every request in the batch, not just this one. A fast 429 with
    Retry-After lets a well-behaved client back off.
    """
    return GatewayError(
        429,
        message or "Server is at capacity. Retry shortly.",
        TYPE_RATE_LIMIT,
        "rate_limit_exceeded",
        headers={"Retry-After": str(max(1, int(retry_after_s)))},
    )


def upstream_unavailable(detail: str) -> GatewayError:
    return GatewayError(
        502, f"Inference engine unreachable: {detail}", TYPE_UPSTREAM, "bad_gateway"
    )


def upstream_timeout(seconds: float) -> GatewayError:
    return GatewayError(
        504,
        f"Inference engine did not respond within {seconds:.0f}s.",
        TYPE_UPSTREAM,
        "timeout",
    )
