"""Pluggable retrieval providers (web search backends)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from dotenv import load_dotenv

from rag.exceptions import RetrievalError
from rag.models import RetrievedDocument

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SERPER_SEARCH_URL = "https://google.serper.dev/search"
DEFAULT_TIMEOUT_SECONDS = 30.0


class RetrievalProvider(ABC):
    """Interface for swapping search/retrieval backends."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        """Return real search results for a query. Must not fabricate content."""


class EmptyRetrievalProvider(RetrievalProvider):
    """Safe default when no API key is configured — returns no results."""

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        logger.warning(
            "No retrieval API key configured; returning empty results. "
            "Set TAVILY_API_KEY or SERPER_API_KEY in .env to enable web search."
        )
        return []


class FixtureRetrievalProvider(RetrievalProvider):
    """Inject fixed documents for local tests — never used in production by default."""

    def __init__(
        self,
        documents: list[RetrievedDocument] | None = None,
        *,
        should_fail: bool = False,
        failure_message: str = "Simulated provider failure",
    ) -> None:
        self._documents = documents or []
        self._should_fail = should_fail
        self._failure_message = failure_message

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        if self._should_fail:
            raise RetrievalError(self._failure_message)
        return self._documents[:top_k]


class TavilyRetrievalProvider(RetrievalProvider):
    """Retrieval via the Tavily Search API."""

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key cannot be empty.")
        self._api_key = api_key.strip()
        self._timeout = timeout

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
            "include_raw_content": False,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(TAVILY_SEARCH_URL, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"Tavily API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise RetrievalError(f"Tavily API request failed: {exc}") from exc

        return _parse_provider_results(data.get("results", []), score_key="score")


class SerperRetrievalProvider(RetrievalProvider):
    """Retrieval via the Serper Google Search API."""

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key.strip():
            raise ValueError("Serper API key cannot be empty.")
        self._api_key = api_key.strip()
        self._timeout = timeout

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedDocument]:
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": top_k}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(SERPER_SEARCH_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"Serper API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise RetrievalError(f"Serper API request failed: {exc}") from exc

        organic = data.get("organic", [])
        documents: list[RetrievedDocument] = []
        for index, item in enumerate(organic[:top_k]):
            title = (item.get("title") or "").strip()
            url = (item.get("link") or "").strip()
            content = (item.get("snippet") or "").strip()
            if not title or not url or not content:
                continue
            position = item.get("position", index + 1)
            score = max(0.0, 1.0 - ((position - 1) * 0.1))
            documents.append(
                RetrievedDocument(
                    title=title,
                    url=url,
                    source=_domain_from_url(url),
                    content=content,
                    relevance_score=round(score, 3),
                )
            )
        return documents


def create_retrieval_provider(
    *,
    provider_name: str | None = None,
    tavily_api_key: str | None = None,
    serper_api_key: str | None = None,
) -> RetrievalProvider:
    """Select a provider from explicit args or environment variables."""
    load_dotenv()

    name = (provider_name or os.getenv("RETRIEVAL_PROVIDER", "")).strip().lower()
    tavily_key = (tavily_api_key or os.getenv("TAVILY_API_KEY") or "").strip()
    serper_key = (serper_api_key or os.getenv("SERPER_API_KEY") or "").strip()

    if name == "empty":
        return EmptyRetrievalProvider()

    if name == "tavily" or (not name and tavily_key):
        if not tavily_key:
            raise ValueError("RETRIEVAL_PROVIDER=tavily requires TAVILY_API_KEY.")
        return TavilyRetrievalProvider(tavily_key)

    if name == "serper" or (not name and serper_key):
        if not serper_key:
            raise ValueError("RETRIEVAL_PROVIDER=serper requires SERPER_API_KEY.")
        return SerperRetrievalProvider(serper_key)

    if name and name not in {"tavily", "serper", "empty"}:
        raise ValueError(
            f"Unknown RETRIEVAL_PROVIDER '{name}'. Use 'tavily', 'serper', or 'empty'."
        )

    return EmptyRetrievalProvider()


def _parse_provider_results(
    results: list[dict[str, Any]],
    *,
    score_key: str,
) -> list[RetrievedDocument]:
    documents: list[RetrievedDocument] = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or item.get("link") or "").strip()
        content = (item.get("content") or item.get("snippet") or "").strip()
        if not title or not url or not content:
            continue

        raw_score = item.get(score_key)
        relevance_score = float(raw_score) if raw_score is not None else None

        documents.append(
            RetrievedDocument(
                title=title,
                url=url,
                source=_domain_from_url(url),
                content=content,
                relevance_score=relevance_score,
            )
        )
    return documents


def _domain_from_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"
