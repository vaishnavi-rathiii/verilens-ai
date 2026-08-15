"""Document and web retrieval for fact-checking queries."""

from __future__ import annotations

import logging

from rag.exceptions import EmptyQueryError, RetrievalError
from rag.models import RetrievedDocument
from rag.providers import RetrievalProvider, create_retrieval_provider
from rag.sources import SourceEntry, enrich_document, get_source_config, rank_documents

logger = logging.getLogger(__name__)

# Backward-compatible re-export for modules that import from retriever.
__all__ = ["RetrievedDocument", "Retriever", "EmptyQueryError", "RetrievalError"]


class Retriever:
    """Fetches relevant documents from a configured retrieval provider."""

    def __init__(
        self,
        provider: RetrievalProvider | None = None,
        source_config: list[SourceEntry] | None = None,
    ) -> None:
        self._provider = provider or create_retrieval_provider()
        self._source_config = get_source_config(source_config)

    @property
    def provider(self) -> RetrievalProvider:
        return self._provider

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        """Return top-k documents for a claim or question.

        Raises:
            EmptyQueryError: If the claim is empty or whitespace-only.
            RetrievalError: If the underlying provider fails.
        """
        claim = query.strip()
        if not claim:
            raise EmptyQueryError("Claim text cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        logger.info("Retrieving documents for claim (top_k=%s)", top_k)

        try:
            raw_documents = self._provider.search(claim, top_k=top_k)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Unexpected retrieval failure: {exc}") from exc

        enriched = [enrich_document(doc, self._source_config) for doc in raw_documents]
        ranked = rank_documents(enriched, self._source_config)
        return ranked[:top_k]
