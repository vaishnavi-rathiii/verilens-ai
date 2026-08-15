"""Tests for evidence extraction."""

from __future__ import annotations

import pytest

from rag.evidence import EvidenceExtractor
from rag.models import EvidenceItem, RetrievedDocument


@pytest.fixture
def extractor() -> EvidenceExtractor:
    return EvidenceExtractor(max_evidence=5, min_relevance=0.15)


def test_extract_relevant_passages_from_documents(extractor: EvidenceExtractor) -> None:
    claim = "COVID vaccines contain microchips for tracking people."
    documents = [
        RetrievedDocument(
            title="PolitiFact: Vaccine microchip claim",
            url="https://www.politifact.com/factchecks/2021/vaccine-microchip/",
            source="PolitiFact",
            content=(
                "Social media posts falsely claim COVID vaccines contain microchips. "
                "Health experts confirm vaccines do not include tracking devices."
            ),
            relevance_score=0.88,
        ),
        RetrievedDocument(
            title="Snopes: Microchip rumor",
            url="https://www.snopes.com/fact-check/vaccine-microchip/",
            source="Snopes",
            content=(
                "The claim that COVID vaccines contain microchips is false. "
                "Independent investigators found no evidence of tracking hardware."
            ),
            relevance_score=0.80,
        ),
    ]

    evidence = extractor.extract(claim, documents)

    assert len(evidence) >= 1
    assert all(isinstance(item, EvidenceItem) for item in evidence)
    assert all("microchip" in item.text.lower() for item in evidence)
    assert evidence[0].source in {"PolitiFact", "Snopes"}
    assert all(item.url.startswith("https://") for item in evidence)
    assert all(item.title for item in evidence)
    assert all(item.relevance_score is not None for item in evidence)
    assert evidence == sorted(evidence, key=lambda item: item.relevance_score or 0.0, reverse=True)


def test_extract_no_documents_returns_empty(extractor: EvidenceExtractor) -> None:
    assert extractor.extract("Any claim text.", []) == []


def test_extract_empty_claim_returns_empty(extractor: EvidenceExtractor) -> None:
    documents = [
        RetrievedDocument(
            title="Example",
            url="https://example.com/post",
            source="example.com",
            content="Some unrelated content about sports and weather.",
            relevance_score=0.5,
        ),
    ]

    assert extractor.extract("", documents) == []
    assert extractor.extract("   ", documents) == []


def test_extract_irrelevant_documents_returns_empty(extractor: EvidenceExtractor) -> None:
    claim = "Mars colony announced by NASA yesterday."
    documents = [
        RetrievedDocument(
            title="Local sports recap",
            url="https://example.com/sports",
            source="example.com",
            content="The basketball team won their season opener last night.",
            relevance_score=0.9,
        ),
    ]

    assert extractor.extract(claim, documents) == []


def test_extract_deduplicates_duplicate_documents(extractor: EvidenceExtractor) -> None:
    claim = "City council approved a new park budget."
    shared_content = (
        "The city council approved a new park budget after public hearings. "
        "Funding will cover playground equipment and green space maintenance."
    )
    documents = [
        RetrievedDocument(
            title="Council vote",
            url="https://news.example.com/council-park",
            source="example.com",
            content=shared_content,
            relevance_score=0.6,
        ),
        RetrievedDocument(
            title="Council vote duplicate",
            url="https://news.example.com/council-park",
            source="example.com",
            content=shared_content,
            relevance_score=0.9,
        ),
    ]

    evidence = extractor.extract(claim, documents)

    assert len(evidence) >= 1
    assert len({item.url for item in evidence}) == len({item.text for item in evidence})


def test_extract_skips_missing_content(extractor: EvidenceExtractor) -> None:
    claim = "Bridge construction delayed until next year."
    valid = RetrievedDocument(
        title="Infrastructure update",
        url="https://example.com/bridge",
        source="example.com",
        content="Bridge construction has been delayed until next year due to supply issues.",
        relevance_score=0.7,
    )
    invalid = RetrievedDocument.model_construct(
        title="Broken record",
        url="https://example.com/broken",
        source="example.com",
        content="",
        relevance_score=0.9,
    )

    evidence = extractor.extract(claim, [invalid, valid])

    assert len(evidence) == 1
    assert evidence[0].url == valid.url
    assert "bridge construction" in evidence[0].text.lower()


def test_extract_respects_max_evidence_limit() -> None:
    claim = "Electric buses reduce urban air pollution significantly."
    documents = [
        RetrievedDocument(
            title=f"Report {index}",
            url=f"https://example.com/report-{index}",
            source="example.com",
            content=(
                f"Report {index}: Electric buses reduce urban air pollution significantly "
                "in major cities according to transit agency studies."
            ),
            relevance_score=0.5 + (index * 0.05),
        )
        for index in range(6)
    ]

    limited_extractor = EvidenceExtractor(max_evidence=3, min_relevance=0.15)
    evidence = limited_extractor.extract(claim, documents)

    assert len(evidence) == 3


def test_extract_preserves_source_attribution(extractor: EvidenceExtractor) -> None:
    claim = "Wildfire smoke affects air quality in the region."
    documents = [
        RetrievedDocument(
            title="Regional air quality alert",
            url="https://www.reuters.com/world/air-quality-alert",
            source="Reuters",
            content="Wildfire smoke affects air quality in the region, officials said.",
            relevance_score=0.75,
        ),
    ]

    evidence = extractor.extract(claim, documents)

    assert len(evidence) == 1
    assert evidence[0].source == "Reuters"
    assert evidence[0].url == "https://www.reuters.com/world/air-quality-alert"
    assert evidence[0].title == "Regional air quality alert"


def test_extract_does_not_invent_facts(extractor: EvidenceExtractor) -> None:
    claim = "Aliens landed in the city center."
    documents = [
        RetrievedDocument(
            title="Traffic update",
            url="https://example.com/traffic",
            source="example.com",
            content="Morning traffic was heavy near the city center.",
            relevance_score=0.4,
        ),
    ]

    evidence = extractor.extract(claim, documents)

    for item in evidence:
        assert "aliens" not in item.text.lower() or "aliens" in documents[0].content.lower()
        assert item.url == documents[0].url


def test_extract_handles_long_document_content(extractor: EvidenceExtractor) -> None:
    claim = "Renewable energy capacity expanded rapidly in 2024."
    long_body = (
        "Unrelated introduction about market trends and quarterly reports. " * 20
        + "Renewable energy capacity expanded rapidly in 2024 across multiple states. "
        + "Additional unrelated commentary about unrelated industries and exports. " * 10
    )
    documents = [
        RetrievedDocument(
            title="Energy market review",
            url="https://example.com/energy-review",
            source="example.com",
            content=long_body,
            relevance_score=0.55,
        ),
    ]

    evidence = extractor.extract(claim, documents)

    assert len(evidence) >= 1
    assert "renewable energy capacity expanded rapidly in 2024" in evidence[0].text.lower()
    assert len(evidence[0].text) <= extractor.max_passage_chars


def test_extract_override_max_evidence_per_call(extractor: EvidenceExtractor) -> None:
    claim = "School district expands free lunch program for students."
    documents = [
        RetrievedDocument(
            title=f"School news {index}",
            url=f"https://example.com/school-{index}",
            source="example.com",
            content=(
                f"Article {index}: The school district expands free lunch program for students "
                "after board approval."
            ),
            relevance_score=0.6,
        )
        for index in range(4)
    ]

    evidence = extractor.extract(claim, documents, max_evidence=2)

    assert len(evidence) == 2
