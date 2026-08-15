"""Evidence extraction from retrieved documents."""

from __future__ import annotations

import logging
import re
from enum import Enum

from rag.models import EvidenceItem, RetrievedDocument
from rag.sources import normalize_url

logger = logging.getLogger(__name__)

DEFAULT_MAX_EVIDENCE = 5
DEFAULT_MIN_RELEVANCE = 0.15
DEFAULT_MAX_PASSAGE_CHARS = 500
DEFAULT_MAX_PASSAGES_PER_DOCUMENT = 2

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "by",
        "as",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
        "yes",
        "about",
        "into",
        "over",
        "after",
        "before",
        "during",
        "than",
        "then",
        "when",
        "where",
        "who",
        "what",
        "which",
        "how",
        "why",
        "if",
        "can",
        "said",
        "says",
    }
)


class EvidenceStance(str, Enum):
    """Reserved for a later step: supporting vs contradicting classification."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class EvidenceExtractor:
    """Extracts relevant passages from retrieved documents for a claim."""

    def __init__(
        self,
        *,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        max_passages_per_document: int = DEFAULT_MAX_PASSAGES_PER_DOCUMENT,
    ) -> None:
        if max_evidence < 1:
            raise ValueError("max_evidence must be at least 1.")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0.0 and 1.0.")

        self.max_evidence = max_evidence
        self.min_relevance = min_relevance
        self.max_passage_chars = max_passage_chars
        self.max_passages_per_document = max_passages_per_document

    def extract(
        self,
        claim: str,
        documents: list[RetrievedDocument],
        *,
        max_evidence: int | None = None,
    ) -> list[EvidenceItem]:
        """Extract ranked evidence passages grounded in retrieved document content."""
        normalized_claim = claim.strip()
        if not normalized_claim:
            logger.warning("Empty claim provided; returning no evidence.")
            return []

        if not documents:
            return []

        limit = max_evidence if max_evidence is not None else self.max_evidence
        if limit < 1:
            raise ValueError("max_evidence must be at least 1.")

        claim_tokens = _tokenize(normalized_claim)
        if not claim_tokens:
            logger.warning("Claim has no usable keywords; returning no evidence.")
            return []

        candidates: list[EvidenceItem] = []
        for document in _dedupe_documents_by_url(documents):
            if not _is_usable_document(document):
                continue

            passages = _extract_passages(document.content, self.max_passage_chars)
            document_candidates: list[tuple[float, EvidenceItem]] = []

            for passage in passages:
                score = _score_passage(claim_tokens, passage, document.relevance_score)
                if score is None or score < self.min_relevance:
                    continue

                document_candidates.append(
                    (
                        score,
                        EvidenceItem(
                            text=passage,
                            source=document.source,
                            url=document.url,
                            title=document.title,
                            relevance_score=score,
                        ),
                    )
                )

            document_candidates.sort(key=lambda item: item[0], reverse=True)
            for _score, item in document_candidates[: self.max_passages_per_document]:
                candidates.append(item)

        ranked = _dedupe_evidence(candidates)
        ranked.sort(
            key=lambda item: item.relevance_score if item.relevance_score is not None else 0.0,
            reverse=True,
        )
        return ranked[:limit]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def _extract_passages(content: str, max_passage_chars: int) -> list[str]:
    """Split document content into candidate passages without altering meaning."""
    normalized = " ".join(content.split())
    if not normalized:
        return []

    if len(normalized) <= max_passage_chars:
        return [normalized]

    sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
    passages: list[str] = []
    buffer = ""

    for part in sentence_parts:
        part = part.strip()
        if not part:
            continue

        candidate = f"{buffer} {part}".strip() if buffer else part
        if len(candidate) <= max_passage_chars:
            buffer = candidate
            continue

        if buffer:
            passages.append(buffer)
        buffer = _truncate_at_word_boundary(part, max_passage_chars)

    if buffer:
        passages.append(buffer)

    if not passages:
        passages.append(_truncate_at_word_boundary(normalized, max_passage_chars))

    return passages


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def _score_passage(
    claim_tokens: set[str],
    passage: str,
    document_relevance: float | None,
) -> float | None:
    passage_tokens = _tokenize(passage)
    if not passage_tokens:
        return None

    overlap = claim_tokens & passage_tokens
    if not overlap:
        return None

    overlap_ratio = len(overlap) / len(claim_tokens)

    if document_relevance is not None:
        combined = (0.65 * overlap_ratio) + (0.35 * document_relevance)
    else:
        combined = overlap_ratio

    return round(min(combined, 1.0), 3)


def _is_usable_document(document: RetrievedDocument) -> bool:
    fields = {
        "content": document.content,
        "url": document.url,
        "title": document.title,
        "source": document.source,
    }
    for name, value in fields.items():
        if not value or not str(value).strip():
            logger.debug("Skipping document with missing %s.", name)
            return False
    return True


def _dedupe_documents_by_url(documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    best_by_url: dict[str, RetrievedDocument] = {}

    for document in documents:
        if not document.url or not document.url.strip():
            continue

        try:
            key = normalize_url(document.url)
        except ValueError:
            key = document.url.strip().lower()

        existing = best_by_url.get(key)
        if existing is None:
            best_by_url[key] = document
            continue

        existing_score = existing.relevance_score if existing.relevance_score is not None else -1.0
        new_score = document.relevance_score if document.relevance_score is not None else -1.0
        if new_score > existing_score:
            best_by_url[key] = document

    return list(best_by_url.values())


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen_text: set[str] = set()
    unique: list[EvidenceItem] = []

    for item in items:
        normalized_text = " ".join(item.text.lower().split())
        if normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)
        unique.append(item)

    return unique
