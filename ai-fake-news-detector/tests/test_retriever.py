"""Tests for the retrieval module."""

from __future__ import annotations

import pytest

from rag.exceptions import EmptyQueryError, RetrievalError
from rag.models import RetrievedDocument
from rag.providers import (
    EmptyRetrievalProvider,
    FixtureRetrievalProvider,
    create_retrieval_provider,
)
from rag.retriever import Retriever
from rag.sources import SourceEntry, rank_documents


@pytest.fixture
def sample_documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            title="Random blog post",
            url="https://example.com/blog/rumor",
            source="example.com",
            content="Unverified social media rumor reposted online.",
            relevance_score=0.95,
        ),
        RetrievedDocument(
            title="PolitiFact rating",
            url="https://www.politifact.com/factchecks/2024/test-claim/",
            source="politifact.com",
            content="Official fact-check conclusion with cited sources.",
            relevance_score=0.70,
        ),
        RetrievedDocument(
            title="Snopes analysis",
            url="https://www.snopes.com/fact-check/test-claim/",
            source="snopes.com",
            content="Investigation summary from Snopes editors.",
            relevance_score=0.65,
        ),
    ]


def test_retriever_returns_ranked_documents(sample_documents: list[RetrievedDocument]) -> None:
    provider = FixtureRetrievalProvider(sample_documents)
    retriever = Retriever(provider=provider)

    results = retriever.retrieve("A politician made a controversial claim.", top_k=3)

    assert len(results) == 3
    assert results[0].source in {"PolitiFact", "Snopes"}
    assert results[0].is_authoritative is True
    assert all(doc.content for doc in results)
    assert all(doc.url.startswith("https://") for doc in results)


def test_retriever_empty_input_raises() -> None:
    retriever = Retriever(provider=EmptyRetrievalProvider())

    with pytest.raises(EmptyQueryError, match="cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_whitespace_only_and_empty_string() -> None:
    retriever = Retriever(provider=EmptyRetrievalProvider())

    with pytest.raises(EmptyQueryError):
        retriever.retrieve("")

    with pytest.raises(EmptyQueryError):
        retriever.retrieve("\n\t")


def test_retriever_provider_failure_raises_retrieval_error() -> None:
    provider = FixtureRetrievalProvider(should_fail=True, failure_message="API unavailable")
    retriever = Retriever(provider=provider)

    with pytest.raises(RetrievalError, match="API unavailable"):
        retriever.retrieve("Test claim about public health.")


def test_empty_provider_returns_no_invented_results() -> None:
    retriever = Retriever(provider=EmptyRetrievalProvider())

    results = retriever.retrieve("Any valid claim text.")

    assert results == []


def test_retriever_respects_top_k(sample_documents: list[RetrievedDocument]) -> None:
    provider = FixtureRetrievalProvider(sample_documents)
    retriever = Retriever(provider=provider)

    results = retriever.retrieve("Claim with multiple sources.", top_k=1)

    assert len(results) == 1


def test_retriever_invalid_top_k_raises() -> None:
    retriever = Retriever(provider=EmptyRetrievalProvider())

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("Valid claim", top_k=0)


def test_rank_documents_prefers_authoritative_sources(
    sample_documents: list[RetrievedDocument],
) -> None:
    ranked = rank_documents(sample_documents)

    authoritative_names = {doc.source for doc in ranked if doc.is_authoritative}
    assert "PolitiFact" in authoritative_names or "Snopes" in authoritative_names
    assert ranked[0].is_authoritative is True


def test_create_provider_empty_mode() -> None:
    provider = create_retrieval_provider(provider_name="empty")
    assert isinstance(provider, EmptyRetrievalProvider)


def test_create_provider_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown RETRIEVAL_PROVIDER"):
        create_retrieval_provider(provider_name="unknown-service")


def test_retrieved_document_rejects_blank_fields() -> None:
    with pytest.raises(ValueError):
        RetrievedDocument(
            title="",
            url="https://example.com",
            source="example.com",
            content="content",
        )


def test_custom_source_config_marks_authority() -> None:
    custom_config = [
        SourceEntry("trusted.example", "Trusted Example", priority=50, authoritative=True),
    ]
    doc = RetrievedDocument(
        title="Trusted report",
        url="https://news.trusted.example/report",
        source="trusted.example",
        content="Report body from configured trusted source.",
        relevance_score=0.5,
    )
    ranked = rank_documents([doc], custom_config)

    assert ranked[0].source == "Trusted Example"
    assert ranked[0].is_authoritative is True
