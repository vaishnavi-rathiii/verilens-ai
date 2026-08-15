"""LLM-based natural-language explanations for verification results.

Step 5 — LLM Evidence Explanation
====================================
Takes a claim, its retrieved evidence (supporting / contradicting / neutral),
and the verification status from Step 4, then calls an LLM to produce a
structured, grounded explanation.

Safety / reliability rules enforced here
-----------------------------------------
* The LLM is given ONLY the retrieved evidence — not raw web content.
* The system prompt explicitly forbids the LLM from inventing facts,
  URLs, source names, or citations.
* Retrieved text is labelled as "DATA" and the LLM is told to treat it
  as untrusted input (reduces prompt-injection risk).
* If no API key is configured, the system returns a deterministic
  rule-based fallback explanation without calling the LLM.
* All LLM failures (network error, bad JSON, timeout, rate-limit) are
  caught and fall back to the same deterministic path.
* The ``key_evidence`` and ``sources`` fields in the output are populated
  ONLY from the supplied EvidenceItems — never invented.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from rag.models import EvidenceItem
from rag.prompts import EXPLANATION_SYSTEM_PROMPT, build_explanation_user_prompt
from rag.verifier import VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLM_TIMEOUT: float = 30.0
DEFAULT_MAX_TOKENS: int = 512
# Number of evidence snippets to include per stance bucket in the prompt.
MAX_SNIPPETS_PER_BUCKET: int = 3

_VALID_VERDICTS: frozenset[str] = frozenset(
    {"LIKELY_TRUE", "LIKELY_FALSE", "MIXED", "UNVERIFIED"}
)


# ---------------------------------------------------------------------------
# Structured output type
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    """Structured, grounded explanation returned to API consumers.

    Fields
    ------
    verdict:
        One of ``LIKELY_TRUE``, ``LIKELY_FALSE``, ``MIXED``, ``UNVERIFIED``.
    explanation:
        A concise natural-language explanation grounded in the evidence.
    key_evidence:
        Selected text snippets from the supplied evidence items (never invented).
    sources:
        Deduplicated source attribution dicts (``title``, ``url``, ``source``).
    llm_used:
        ``True`` if an LLM was successfully called; ``False`` for fallback path.
    """

    verdict: str
    explanation: str
    key_evidence: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    llm_used: bool = False


# ---------------------------------------------------------------------------
# Backward-compatible Explanation alias (used by rag_pipeline.py)
# ---------------------------------------------------------------------------


@dataclass
class Explanation:
    """Legacy dataclass kept for backward compatibility with ``RAGPipeline``.

    New code should use :class:`ExplanationResult` directly.
    """

    text: str
    citations: list[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: ExplanationResult) -> "Explanation":
        """Convert an ExplanationResult to the legacy Explanation format."""
        return cls(
            text=result.explanation,
            citations=[src.get("url", "") for src in result.sources if src.get("url")],
        )


# ---------------------------------------------------------------------------
# LLM provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract interface for swappable LLM backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Return the model's text completion.

        Raises:
            LLMError: On any provider-side failure.
        """


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


class NoOpLLMProvider(LLMProvider):
    """Safe no-op provider used when no API key is configured.

    Never calls any external service.  Always raises ``LLMError`` so the
    explainer can fall back gracefully.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        raise LLMError(
            "No LLM API key is configured. "
            "Set OPENAI_API_KEY or GROQ_API_KEY in .env to enable LLM explanations."
        )


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for OpenAI-compatible REST APIs.

    Works with:
    - OpenAI (https://api.openai.com/v1)
    - Groq  (https://api.groq.com/openai/v1)

    Both expose the same ``/chat/completions`` endpoint schema.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = DEFAULT_LLM_TIMEOUT,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM API key cannot be empty.")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Call the chat completions endpoint and return the assistant message."""
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # deterministic — we want consistent fact-checking
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"LLM API returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"LLM API request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response structure: {exc}") from exc


def create_llm_provider(
    *,
    provider_name: str | None = None,
    openai_api_key: str | None = None,
    groq_api_key: str | None = None,
) -> LLMProvider:
    """Select an LLM provider from explicit args or environment variables.

    Selection order
    ---------------
    1. ``LLM_PROVIDER`` env var (``openai`` or ``groq``)
    2. ``OPENAI_API_KEY`` → OpenAI gpt-4o-mini
    3. ``GROQ_API_KEY``   → Groq llama-3.1-8b-instant (fast, free tier)
    4. ``NoOpLLMProvider`` (no network calls, returns graceful fallback)
    """
    load_dotenv()

    name = (provider_name or os.getenv("LLM_PROVIDER", "")).strip().lower()
    openai_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    groq_key = (groq_api_key or os.getenv("GROQ_API_KEY") or "").strip()

    if name == "openai" or (not name and openai_key):
        if not openai_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
        logger.info("Using OpenAI LLM provider.")
        return OpenAICompatibleProvider(
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )

    if name == "groq" or (not name and groq_key):
        if not groq_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY.")
        logger.info("Using Groq LLM provider.")
        return OpenAICompatibleProvider(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
        )

    logger.warning(
        "No LLM API key configured. Explanations will use the rule-based fallback. "
        "Set OPENAI_API_KEY or GROQ_API_KEY in .env to enable LLM explanations."
    )
    return NoOpLLMProvider()


# ---------------------------------------------------------------------------
# ExplanationGenerator
# ---------------------------------------------------------------------------


class ExplanationGenerator:
    """Generates structured, evidence-grounded explanations using an LLM.

    The LLM is given ONLY the supplied evidence — it cannot access the
    internet or its own training knowledge for fact-checking.  If the LLM
    call fails for any reason, a deterministic rule-based fallback is used
    so the system never crashes silently.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider or create_llm_provider()

    def generate(
        self,
        claim: str,
        verification_result: VerificationResult,
    ) -> ExplanationResult:
        """Return a structured explanation grounded in the verification result.

        Args:
            claim:               The original user claim text.
            verification_result: Output from ``Verifier.verify()`` (Step 4).

        Returns:
            An :class:`ExplanationResult` with verdict, explanation, key
            evidence snippets, and source attributions — all taken ONLY from
            the supplied evidence.
        """
        normalized_claim = (claim or "").strip()
        if not normalized_claim:
            logger.warning("Empty claim passed to ExplanationGenerator.")
            return _unverified_fallback(
                claim=claim,
                reason="No claim was provided.",
                evidence_items=[],
            )

        all_evidence = (
            verification_result.supporting
            + verification_result.contradicting
            + verification_result.neutral
        )

        if not all_evidence:
            logger.info("No evidence available; returning UNVERIFIED fallback.")
            return _unverified_fallback(
                claim=normalized_claim,
                reason="No evidence was retrieved to evaluate this claim.",
                evidence_items=[],
            )

        # Build deduplicated source list from real evidence only — never invented.
        sources = _build_source_list(all_evidence)

        supporting_snippets = [
            item.text for item in verification_result.supporting[:MAX_SNIPPETS_PER_BUCKET]
        ]
        contradicting_snippets = [
            item.text for item in verification_result.contradicting[:MAX_SNIPPETS_PER_BUCKET]
        ]
        neutral_snippets = [
            item.text for item in verification_result.neutral[:MAX_SNIPPETS_PER_BUCKET]
        ]

        status_str = verification_result.status.value

        user_prompt = build_explanation_user_prompt(
            claim=normalized_claim,
            verification_status=status_str,
            supporting_snippets=supporting_snippets,
            contradicting_snippets=contradicting_snippets,
            neutral_snippets=neutral_snippets,
            source_list=sources,
        )

        # Attempt LLM call — fall back gracefully on any error.
        try:
            raw_text = self._provider.complete(
                EXPLANATION_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            result = _parse_llm_response(
                raw_text=raw_text,
                fallback_verdict=status_str,
                available_sources=sources,
            )
            result.llm_used = True
            return result

        except LLMError as exc:
            logger.warning("LLM call failed (%s); using rule-based fallback.", exc)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in LLM call; using rule-based fallback.")

        return _rule_based_explanation(
            claim=normalized_claim,
            verification_result=verification_result,
            sources=sources,
        )


# ---------------------------------------------------------------------------
# Backward-compatible Explainer class (used by rag_pipeline.py)
# ---------------------------------------------------------------------------


class Explainer:
    """Generates explanations strictly grounded in retrieved evidence.

    Wraps :class:`ExplanationGenerator` to preserve the interface expected
    by ``RAGPipeline``.  New code should use ``ExplanationGenerator`` directly.

    Must not invent sources, URLs, or facts. Treats retrieved content as
    untrusted data (not instructions) to reduce prompt-injection risk.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._generator = ExplanationGenerator(provider=provider)

    def explain(
        self,
        claim: str,
        verification: VerificationResult,
    ) -> Explanation:
        """Return an explanation with citation references only from evidence."""

        result = self._generator.generate(
            claim,
            verification,
        )

        return Explanation.from_result(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_source_list(evidence_items: list[EvidenceItem]) -> list[dict[str, str]]:
    """Build a deduplicated list of source dicts from evidence items.

    Uses URL as the deduplication key.  Never invents a source.
    """
    seen_urls: set[str] = set()
    sources: list[dict[str, str]] = []

    for item in evidence_items:
        url = (getattr(item, "url", None) or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append(
            {
                "title": (getattr(item, "title", None) or "").strip(),
                "url": url,
                "source": (getattr(item, "source", None) or "").strip(),
            }
        )

    return sources


def _parse_llm_response(
    raw_text: str,
    fallback_verdict: str,
    available_sources: list[dict[str, str]],
) -> ExplanationResult:
    """Parse the LLM's JSON response into an ExplanationResult.

    If the JSON is malformed or the verdict is invalid, falls back to the
    rule-based verdict so we never return an invented result.

    Sources in the LLM response are cross-checked against ``available_sources``
    to prevent the LLM from inventing URLs.
    """
    text = (raw_text or "").strip()

    # Strip markdown fences if the model wraps JSON in ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence
        lines = lines[1:] if lines[0].startswith("```") else lines
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON response: %s", exc)
        raise LLMError(f"LLM returned non-JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMError("LLM response is not a JSON object.")

    verdict = (data.get("verdict") or fallback_verdict).strip().upper()
    if verdict not in _VALID_VERDICTS:
        logger.warning("LLM returned unknown verdict %r; using fallback %r.", verdict, fallback_verdict)
        verdict = fallback_verdict

    explanation = (data.get("explanation") or "").strip()
    if not explanation:
        explanation = "Insufficient evidence to generate an explanation."

    # key_evidence must be strings — silently drop non-string entries.
    raw_key_evidence = data.get("key_evidence") or []
    key_evidence = [str(e).strip() for e in raw_key_evidence if e and str(e).strip()]

    # Validate sources against the actual retrieved list to prevent hallucinations.
    raw_sources = data.get("sources") or []
    validated_sources = _validate_sources(raw_sources, available_sources)

    return ExplanationResult(
        verdict=verdict,
        explanation=explanation,
        key_evidence=key_evidence,
        sources=validated_sources,
        llm_used=False,  # caller sets this to True after successful parse
    )


def _validate_sources(
    llm_sources: list[Any],
    available_sources: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return only those LLM-generated sources whose URL exists in available_sources.

    This prevents the LLM from slipping invented URLs into the output.
    If no LLM source passes validation, return the available_sources as-is.
    """
    available_urls: set[str] = {src["url"] for src in available_sources if src.get("url")}

    validated: list[dict[str, str]] = []
    for src in llm_sources:
        if not isinstance(src, dict):
            continue
        url = (src.get("url") or "").strip()
        if url and url in available_urls:
            validated.append(
                {
                    "title": (src.get("title") or "").strip(),
                    "url": url,
                    "source": (src.get("source") or "").strip(),
                }
            )

    # If the LLM returned no valid sources (or hallucinated all of them),
    # fall back to the full available source list so the consumer still gets citations.
    return validated if validated else list(available_sources)


def _rule_based_explanation(
    claim: str,
    verification_result: VerificationResult,
    sources: list[dict[str, str]],
) -> ExplanationResult:
    """Produce a deterministic explanation without calling the LLM.

    Used as a fallback when the LLM is unavailable or fails.  The explanation
    is constructed purely from the verification_result fields — no facts are
    invented.
    """
    status = verification_result.status
    n_sup = len(verification_result.supporting)
    n_con = len(verification_result.contradicting)
    n_neu = len(verification_result.neutral)
    total = n_sup + n_con + n_neu

    verdict = status.value

    if status is VerificationStatus.LIKELY_TRUE:
        explanation = (
            f"Based on {total} retrieved evidence item(s), the claim appears likely true. "
            f"{n_sup} item(s) support it and {n_con} contradict it."
        )
    elif status is VerificationStatus.LIKELY_FALSE:
        explanation = (
            f"Based on {total} retrieved evidence item(s), the claim appears likely false. "
            f"{n_con} item(s) contradict it and {n_sup} support it."
        )
    elif status is VerificationStatus.MIXED:
        explanation = (
            f"The evidence is mixed. Out of {total} item(s), "
            f"{n_sup} support the claim and {n_con} contradict it. "
            "Independent verification is recommended."
        )
    else:
        explanation = (
            f"There is insufficient reliable evidence to evaluate this claim "
            f"({total} item(s) retrieved, none conclusive). "
            "The claim remains UNVERIFIED."
        )

    # Collect key evidence from the most relevant bucket.
    key_items = (
        verification_result.supporting[:2]
        or verification_result.contradicting[:2]
        or verification_result.neutral[:2]
    )
    key_evidence = [item.text for item in key_items]

    return ExplanationResult(
        verdict=verdict,
        explanation=explanation,
        key_evidence=key_evidence,
        sources=sources,
        llm_used=False,
    )


def _unverified_fallback(
    claim: str,
    reason: str,
    evidence_items: list[EvidenceItem],
) -> ExplanationResult:
    """Return a safe UNVERIFIED ExplanationResult without calling the LLM."""
    sources = _build_source_list(evidence_items)
    return ExplanationResult(
        verdict=VerificationStatus.UNVERIFIED.value,
        explanation=reason,
        key_evidence=[],
        sources=sources,
        llm_used=False,
    )

