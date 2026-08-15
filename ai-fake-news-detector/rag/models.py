"""Shared data models for the RAG module."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetrievedDocument(BaseModel):
    """A single document returned by a retrieval provider."""

    title: str
    url: str
    source: str
    content: str
    relevance_score: float | None = None
    is_authoritative: bool = False

    model_config = {"frozen": True}

    @field_validator("title", "url", "source", "content")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be blank.")
        return value.strip()

    @field_validator("relevance_score")
    @classmethod
    def score_in_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("relevance_score must be between 0.0 and 1.0.")
        return value


class EvidenceItem(BaseModel):
    """A ranked evidence passage extracted from a retrieved document."""

    text: str
    source: str
    url: str
    title: str
    relevance_score: float | None = None

    model_config = {"frozen": True}

    @field_validator("text", "source", "url", "title")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be blank.")
        return value.strip()

    @field_validator("relevance_score")
    @classmethod
    def score_in_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("relevance_score must be between 0.0 and 1.0.")
        return value
