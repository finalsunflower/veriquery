"""
Unified error handling module for the VeriQuery API.

Registers exception handlers with FastAPI to ensure all errors — validation
failures, HTTP exceptions, business errors, and unexpected exceptions — are
returned as standardized ErrorResponse JSON payloads.

Handler priority (specific to general):
  1. RequestValidationError  → 422 (parameter validation failure)
  2. StarletteHTTPException  → original status code (404, 403, etc.)
  3. VeriQueryError          → 503 or 500 depending on subclass
     ├─ ConfigurationError   → 503 (service unavailable, recoverable)
     ├─ RetrievalError       → 503 (external dependency unavailable)
     └─ ProcessingError      → 500 (internal processing failure)
  4. Exception               → 500 (catch-all, details only in DEBUG mode)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import VeriQueryError, ConfigurationError, RetrievalError, ProcessingError

logger = logging.getLogger(__name__)


class ValidationErrorDetail(BaseModel):
    """Single field validation error detail.

    Attributes:
        field: Dot-separated field path (e.g. "body.query").
        message: Validation error message from Pydantic.
        value: The invalid value submitted by the user, if available.
    """
    field: str
    message: str
    value: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Unified error response model.

    All exception handlers produce this structure so the frontend can
    parse errors consistently regardless of the error type.

    Attributes:
        success: Always False for error responses.
        error: Human-readable error description.
        error_code: Machine-readable code (e.g. "VALIDATION_ERROR", "HTTP_404").
        details: Optional context dict with additional error information.
        errors: Optional list of field-level validation errors.
        timestamp: ISO 8601 timestamp of when the error occurred.
    """
    success: bool = False
    error: str = ""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    errors: Optional[List[ValidationErrorDetail]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request parameter validation failures (422).

    Triggered before the route handler executes when Pydantic validation
    fails on request parameters.

    Args:
        request: Current request object (injected by FastAPI).
        exc: Validation exception with field-level error details.

    Returns:
        JSONResponse with status 422 and ErrorResponse body.
    """
    logger.warning(f"Validation Error: {exc.errors()}")

    errors = [
        ValidationErrorDetail(
            field=".".join(str(loc) for loc in error["loc"]),
            message=error["msg"],
            value=error.get("input")
        )
        for error in exc.errors()
    ]

    response = ErrorResponse(
        error="Validation failed",
        error_code="VALIDATION_ERROR",
        errors=errors
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response.model_dump(exclude_none=True)
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP protocol-level exceptions (404, 403, etc.).

    Must register StarletteHTTPException (not fastapi.HTTPException) to
    intercept all HTTP errors including those raised internally by FastAPI
    (e.g. 404 route not found).

    Args:
        request: Current request object (injected by FastAPI).
        exc: Starlette HTTP exception with status_code and detail.

    Returns:
        JSONResponse with the original HTTP status code and ErrorResponse body.
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")

    response = ErrorResponse(
        error=str(exc.detail),
        error_code=f"HTTP_{exc.status_code}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(exclude_none=True)
    )


async def veriquery_error_handler(request: Request, exc: VeriQueryError) -> JSONResponse:
    """Handle VeriQuery custom business exceptions.

    Maps exception subclasses to appropriate HTTP status codes:
        ConfigurationError → 503 (external dependency, recoverable)
        RetrievalError     → 503 (external dependency, recoverable)
        ProcessingError    → 500 (internal logic failure)
        VeriQueryError     → 500 (fallback for unclassified business errors)

    Args:
        request: Current request object (injected by FastAPI).
        exc: VeriQueryError or subclass with message, code, and optional details.

    Returns:
        JSONResponse with mapped status code and ErrorResponse body.
    """
    if isinstance(exc, ConfigurationError):
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, RetrievalError):
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, ProcessingError):
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR

    logger.error(f"Business Error [{exc.code}]: {exc.message}")

    response = ErrorResponse(
        error=exc.message,
        error_code=exc.code,
        details=exc.details if exc.details else None,
    )

    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(exclude_none=True),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Returns a generic 500 response. Exception details are only included
    when running in DEBUG log level to prevent sensitive information
    leakage (DB connection strings, file paths, API keys, etc.) in
    production environments.

    Args:
        request: Current request object (injected by FastAPI).
        exc: Any unhandled Python exception.

    Returns:
        JSONResponse with status 500 and ErrorResponse body.
    """
    logger.error("Unhandled Exception: %s", exc, exc_info=True)

    response = ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_SERVER_ERROR",
        details={"detail": str(exc)} if logger.isEnabledFor(logging.DEBUG) else None
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(exclude_none=True)
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application.

    Call this once after creating the FastAPI instance in main.py.
    Registration order does not affect matching — FastAPI selects the
    most specific handler based on exception type inheritance.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(VeriQueryError, veriquery_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
