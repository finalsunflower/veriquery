"""
Unified exception hierarchy for the VeriQuery system.

All custom business exceptions inherit from VeriQueryError, providing a
consistent (message + code + details) triplet structure. Subclasses map
to semantic HTTP status codes via api/error_handlers.py:

  - ConfigurationError → 503 Service Unavailable
  - RetrievalError     → 503 Service Unavailable
  - ProcessingError    → 500 Internal Server Error
  - VeriQueryError     → 500 Internal Server Error (fallback)
"""

from typing import Dict, Any


class VeriQueryError(Exception):
    """Base exception for all VeriQuery business errors.

    Attributes:
        message: Human-readable error description.
        details: Structured context for programmatic consumption.
        code: Error code for log search and i18n (defaults to class name).
    """

    def __init__(
        self,
        message: str,
        details: Dict[str, Any] = None,
        code: str = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.code = code or self.__class__.__name__


class RetrievalError(VeriQueryError):
    """Raised when vector or keyword retrieval fails.

    Typically caused by external service unavailability (ChromaDB, SQLite).
    Maps to HTTP 503 in error_handlers.py.
    """
    pass


class ConfigurationError(VeriQueryError):
    """Raised when system configuration is invalid or missing.

    Maps to HTTP 503 — recoverable once configuration is fixed.
    """
    pass


class ProcessingError(VeriQueryError):
    """Raised when document or data processing fails.

    Maps to HTTP 500 — typically indicates a logic error, not a transient issue.
    """
    pass
