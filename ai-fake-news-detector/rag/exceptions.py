"""Custom exceptions for retrieval and RAG operations."""


class RetrievalError(Exception):
    """Raised when a retrieval provider fails."""


class EmptyQueryError(ValueError):
    """Raised when the user claim/query is empty or whitespace-only."""
