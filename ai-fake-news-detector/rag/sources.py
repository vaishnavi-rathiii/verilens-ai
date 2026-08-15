"""Source configuration, priority logic, and citation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from rag.models import EvidenceItem, RetrievedDocument


@dataclass(frozen=True)
class SourceEntry:
    """Configurable source metadata.

    A source is only treated as authoritative when ``authoritative=True`` here.
    """

    domain: str
    name: str
    priority: int = 0
    authoritative: bool = False


DEFAULT_SOURCE_ENTRIES: tuple[SourceEntry, ...] = (
    SourceEntry("snopes.com", "Snopes", priority=100, authoritative=True),
    SourceEntry("politifact.com", "PolitiFact", priority=100, authoritative=True),
    SourceEntry("factcheck.org", "FactCheck.org", priority=100, authoritative=True),
    SourceEntry("apnews.com", "Associated Press", priority=90, authoritative=True),
    SourceEntry("reuters.com", "Reuters", priority=90, authoritative=True),
    SourceEntry("bbc.com", "BBC", priority=80, authoritative=False),
    SourceEntry("nytimes.com", "The New York Times", priority=70, authoritative=False),
    SourceEntry("who.int", "World Health Organization", priority=85, authoritative=True),
    SourceEntry("cdc.gov", "Centers for Disease Control", priority=85, authoritative=True),
)


@dataclass(frozen=True)
class SourceCitation:
    """Normalized citation returned to API consumers."""

    url: str
    title: str | None = None
    publisher: str | None = None


def get_source_config(
    entries: tuple[SourceEntry, ...] | list[SourceEntry] | None = None,
) -> list[SourceEntry]:
    """Return the active source configuration list."""
    if entries is None:
        return list(DEFAULT_SOURCE_ENTRIES)
    return list(entries)


def extract_domain(url: str) -> str:
    """Extract a normalized domain from a URL."""
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        raise ValueError(f"Invalid URL: {url}")
    return host


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication and display."""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    return parsed.geturl()


def match_source(url: str, source_config: list[SourceEntry]) -> SourceEntry | None:
    """Match a URL to the best configured source entry, if any."""
    domain = extract_domain(url)
    best_match: SourceEntry | None = None

    for entry in source_config:
        entry_domain = entry.domain.lower()
        if domain == entry_domain or domain.endswith(f".{entry_domain}"):
            if best_match is None or entry.priority > best_match.priority:
                best_match = entry

    return best_match


def resolve_source_metadata(
    url: str,
    source_config: list[SourceEntry],
) -> tuple[str, bool, int]:
    """Return display name, authoritative flag, and priority for a URL."""
    matched = match_source(url, source_config)
    if matched:
        return matched.name, matched.authoritative, matched.priority
    return extract_domain(url), False, 0


def enrich_document(
    document: RetrievedDocument,
    source_config: list[SourceEntry],
) -> RetrievedDocument:
    """Attach configured source name and authoritative flag to a document."""
    name, authoritative, _priority = resolve_source_metadata(document.url, source_config)
    updates: dict[str, object] = {"source": name, "is_authoritative": authoritative}
    return document.model_copy(update=updates)


def rank_documents(
    documents: list[RetrievedDocument],
    source_config: list[SourceEntry] | None = None,
) -> list[RetrievedDocument]:
    """Rank documents by configured authority, priority, then relevance score."""
    config = get_source_config(source_config)

    def sort_key(doc: RetrievedDocument) -> tuple[int, int, float]:
        _name, authoritative, priority = resolve_source_metadata(doc.url, config)
        score = doc.relevance_score if doc.relevance_score is not None else 0.0
        return (1 if authoritative else 0, priority, score)

    enriched = [enrich_document(doc, config) for doc in documents]
    return sorted(enriched, key=sort_key, reverse=True)


def build_citations(
    evidence: list[EvidenceItem],
) -> list[SourceCitation]:
    """Build deduplicated citations from evidence items."""

    seen: set[str] = set()
    citations: list[SourceCitation] = []

    for item in evidence:
        url = item.url.strip()

        if not url:
            continue

        try:
            normalized = normalize_url(url)
        except ValueError:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        citations.append(
            SourceCitation(
                url=normalized,
                title=item.title,
                publisher=item.source,
            )
        )

    return citations
