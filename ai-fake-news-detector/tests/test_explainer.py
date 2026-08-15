"""Tests for the LLM-based explainer (Step 5 — Evidence Explanation).

All tests are fully mockable and do NOT require a real API key.

Covered scenarios
-----------------
1.  LIKELY_TRUE verdict   — LLM returns valid supporting JSON
2.  LIKELY_FALSE verdict  — LLM returns valid contradicting JSON
3.  MIXED verdict         — LLM returns mixed JSON
4.  UNVERIFIED verdict    — LLM returns UNVERIFIED JSON
5.  Empty evidence        — ExplanationGenerator returns UNVERIFIED fallback
6.  Missing API key       — NoOpLLMProvider triggers rule-based fallback
7.  LLM network failure   — LLMError triggers rule-based fallback
8.  Invalid LLM response  — Non-JSON / bad verdict triggers fallback
9.  Source validation     — LLM-invented URLs are stripped from output
10. Markdown fence stripping — ```json...``` responses are parsed correctly
11. Empty claim           — Returns UNVERIFIED immediately
12. Backward-compatible Explainer.explain() — returns Explanation dataclass
13. _build_source_list    — deduplicates by URL, never invents
14. _validate_sources     — rejects URLs not in available set
15. _rule_based_explanation — each verdict branch produces correct text
16. ExplanationResult.llm_used flag — True when LLM succeeds
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.explainer import (
    Explainer,
    ExplanationGenerator,
    ExplanationResult,
    LLMError,
    LLMProvider,
    NoOpLLMProvider,
    OpenAICompatibleProvider,
    _build_source_list,
    _parse_llm_response,
    _rule_based_explanation,
    _validate_sources,
    create_llm_provider,
)
from rag.models import EvidenceItem
from rag.verifier import VerificationResult, VerificationStatus


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _item(
    text: str,
    *,
    url: str = "https://example.com/article",
    source: str = "example.com",
    title: str = "Test Article",
    relevance_score: float = 0.75,
) -> EvidenceItem:
    return EvidenceItem(
        text=text,
        url=url,
        source=source,
        title=title,
        relevance_score=relevance_score,
    )


def _make_verification_result(
    status: VerificationStatus,
    supporting: list[EvidenceItem] | None = None,
    contradicting: list[EvidenceItem] | None = None,
    neutral: list[EvidenceItem] | None = None,
    summary: str | None = None,
) -> VerificationResult:
    return VerificationResult(
        status=status,
        supporting=supporting or [],
        contradicting=contradicting or [],
        neutral=neutral or [],
        summary=summary,
    )


def _valid_llm_json(
    verdict: str = "LIKELY_TRUE",
    explanation: str = "Evidence confirms the claim.",
    key_evidence: list[str] | None = None,
    sources: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "explanation": explanation,
            "key_evidence": key_evidence or ["Studies confirm this."],
            "sources": sources
            or [
                {
                    "title": "Test Article",
                    "url": "https://example.com/article",
                    "source": "example.com",
                }
            ],
        }
    )


class StubLLMProvider(LLMProvider):
    """Test stub that returns a fixed string without any network call."""

    def __init__(self, response: str | None = None, should_fail: bool = False) -> None:
        self._response = response or "{}"
        self._should_fail = should_fail

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        if self._should_fail:
            raise LLMError("Simulated LLM failure.")
        return self._response


@pytest.fixture
def supporting_item() -> EvidenceItem:
    return _item("Studies confirm that vaccines are safe and effective.")


@pytest.fixture
def contradicting_item() -> EvidenceItem:
    return _item(
        "The claim is false. Experts debunked this assertion.",
        url="https://snopes.com/fact-check/test",
        source="Snopes",
        title="Snopes Fact Check",
    )


@pytest.fixture
def neutral_item() -> EvidenceItem:
    return _item("Scientists gathered at a symposium to discuss vaccine research.")


# ---------------------------------------------------------------------------
# 1. LIKELY_TRUE — LLM returns valid JSON
# ---------------------------------------------------------------------------


def test_likely_true_with_llm_response(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(
        response=_valid_llm_json(
            verdict="LIKELY_TRUE",
            explanation="The evidence confirms the claim is accurate.",
        )
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Vaccines are safe.", vr)

    assert result.verdict == "LIKELY_TRUE"
    assert "confirm" in result.explanation.lower() or len(result.explanation) > 0
    assert result.llm_used is True
    assert isinstance(result.key_evidence, list)
    assert isinstance(result.sources, list)


# ---------------------------------------------------------------------------
# 2. LIKELY_FALSE — LLM returns valid JSON
# ---------------------------------------------------------------------------


def test_likely_false_with_llm_response(contradicting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_FALSE, contradicting=[contradicting_item]
    )
    provider = StubLLMProvider(
        response=_valid_llm_json(
            verdict="LIKELY_FALSE",
            explanation="Evidence contradicts the claim.",
            sources=[
                {
                    "title": "Snopes Fact Check",
                    "url": "https://snopes.com/fact-check/test",
                    "source": "Snopes",
                }
            ],
        )
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("The vaccine contains microchips.", vr)

    assert result.verdict == "LIKELY_FALSE"
    assert result.llm_used is True
    assert len(result.sources) >= 1


# ---------------------------------------------------------------------------
# 3. MIXED — LLM returns valid JSON
# ---------------------------------------------------------------------------


def test_mixed_verdict_with_llm_response(
    supporting_item: EvidenceItem, contradicting_item: EvidenceItem
) -> None:
    vr = _make_verification_result(
        VerificationStatus.MIXED,
        supporting=[supporting_item],
        contradicting=[contradicting_item],
    )
    provider = StubLLMProvider(
        response=_valid_llm_json(verdict="MIXED", explanation="Evidence is split.")
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Vaccines are 100% risk-free.", vr)

    assert result.verdict == "MIXED"
    assert result.llm_used is True


# ---------------------------------------------------------------------------
# 4. UNVERIFIED — LLM returns valid JSON
# ---------------------------------------------------------------------------


def test_unverified_with_llm_response(neutral_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.UNVERIFIED, neutral=[neutral_item]
    )
    provider = StubLLMProvider(
        response=_valid_llm_json(
            verdict="UNVERIFIED",
            explanation="Insufficient evidence to reach a conclusion.",
        )
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Aliens landed in Texas.", vr)

    assert result.verdict == "UNVERIFIED"
    assert result.llm_used is True


# ---------------------------------------------------------------------------
# 5. Empty evidence → UNVERIFIED fallback (no LLM call)
# ---------------------------------------------------------------------------


def test_empty_evidence_returns_unverified_no_llm() -> None:
    vr = _make_verification_result(VerificationStatus.UNVERIFIED)
    provider = StubLLMProvider(response=_valid_llm_json("LIKELY_TRUE"))
    gen = ExplanationGenerator(provider=provider)

    result = gen.generate("Any claim.", vr)

    assert result.verdict == "UNVERIFIED"
    assert result.llm_used is False
    assert result.sources == []
    assert result.key_evidence == []


# ---------------------------------------------------------------------------
# 6. Missing API key → NoOpLLMProvider → rule-based fallback
# ---------------------------------------------------------------------------


def test_noop_provider_triggers_rule_based_fallback(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    gen = ExplanationGenerator(provider=NoOpLLMProvider())
    result = gen.generate("Vaccines are effective.", vr)

    assert result.verdict == "LIKELY_TRUE"
    assert result.llm_used is False
    assert len(result.explanation) > 0
    assert len(result.sources) >= 1


# ---------------------------------------------------------------------------
# 7. LLM network failure → rule-based fallback
# ---------------------------------------------------------------------------


def test_llm_failure_falls_back_to_rule_based(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    failing_provider = StubLLMProvider(should_fail=True)
    gen = ExplanationGenerator(provider=failing_provider)
    result = gen.generate("Vaccines are effective.", vr)

    assert result.verdict == "LIKELY_TRUE"
    assert result.llm_used is False
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# 8. Invalid LLM response (non-JSON, bad verdict) → fallback
# ---------------------------------------------------------------------------


def test_non_json_llm_response_falls_back(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(response="This is not JSON at all.")
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Claim text.", vr)

    assert result.verdict == "LIKELY_TRUE"
    assert result.llm_used is False


def test_unknown_verdict_from_llm_uses_fallback_verdict(supporting_item: EvidenceItem) -> None:
    """LLM returning an invalid verdict value should be replaced with the rule-based status."""
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(
        response=json.dumps(
            {
                "verdict": "TOTALLY_MADE_UP",
                "explanation": "Something.",
                "key_evidence": [],
                "sources": [],
            }
        )
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Claim text.", vr)

    # LLM parse succeeds (valid JSON) but bad verdict → replaced with rule-based
    assert result.verdict == "LIKELY_TRUE"


# ---------------------------------------------------------------------------
# 9. Source validation — LLM-invented URLs are rejected
# ---------------------------------------------------------------------------


def test_invented_sources_are_stripped_from_output(supporting_item: EvidenceItem) -> None:
    """Sources the LLM invents (URLs not in evidence) must not appear in the output."""
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    # LLM returns a source with an invented URL
    provider = StubLLMProvider(
        response=json.dumps(
            {
                "verdict": "LIKELY_TRUE",
                "explanation": "Evidence confirms.",
                "key_evidence": ["Studies confirm this."],
                "sources": [
                    {
                        "title": "INVENTED ARTICLE",
                        "url": "https://totally-fake-invented.io/article",
                        "source": "Fake Outlet",
                    }
                ],
            }
        )
    )
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Vaccines are safe.", vr)

    # The invented URL must not be in the output; real evidence URL must be there
    output_urls = {src["url"] for src in result.sources}
    assert "https://totally-fake-invented.io/article" not in output_urls
    assert "https://example.com/article" in output_urls


# ---------------------------------------------------------------------------
# 10. Markdown fence stripping
# ---------------------------------------------------------------------------


def test_markdown_fenced_json_is_parsed_correctly(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    fenced_response = (
        "```json\n"
        + _valid_llm_json(verdict="LIKELY_TRUE", explanation="The claim is supported.")
        + "\n```"
    )
    provider = StubLLMProvider(response=fenced_response)
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Vaccines are effective.", vr)

    assert result.verdict == "LIKELY_TRUE"
    assert result.llm_used is True


# ---------------------------------------------------------------------------
# 11. Empty claim → UNVERIFIED immediately
# ---------------------------------------------------------------------------


def test_empty_claim_returns_unverified(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(response=_valid_llm_json("LIKELY_TRUE"))
    gen = ExplanationGenerator(provider=provider)

    for bad_claim in ("", "   ", "\t"):
        result = gen.generate(bad_claim, vr)
        assert result.verdict == "UNVERIFIED", f"Expected UNVERIFIED for {bad_claim!r}"
        assert result.llm_used is False


# ---------------------------------------------------------------------------
# 12. Backward-compatible Explainer.explain()
# ---------------------------------------------------------------------------


def test_explainer_explain_returns_explanation_dataclass(supporting_item: EvidenceItem) -> None:
    from rag.explainer import Explanation

    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(response=_valid_llm_json("LIKELY_TRUE"))
    explainer = Explainer(provider=provider)
    explanation = explainer.explain("Vaccines are safe.", vr)

    assert isinstance(explanation, Explanation)
    assert isinstance(explanation.text, str)
    assert len(explanation.text) > 0
    assert isinstance(explanation.citations, list)


# ---------------------------------------------------------------------------
# 13. _build_source_list — deduplication
# ---------------------------------------------------------------------------


def test_build_source_list_deduplicates_by_url() -> None:
    items = [
        _item("Text A", url="https://example.com/a"),
        _item("Text B", url="https://example.com/a"),  # duplicate URL
        _item("Text C", url="https://example.com/b"),
    ]
    sources = _build_source_list(items)

    assert len(sources) == 2
    urls = [s["url"] for s in sources]
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls


def test_build_source_list_empty_input() -> None:
    assert _build_source_list([]) == []


def test_build_source_list_preserves_source_attribution() -> None:
    item = _item(
        "Some text",
        url="https://politifact.com/fact-check/test",
        source="PolitiFact",
        title="PolitiFact Check",
    )
    sources = _build_source_list([item])

    assert sources[0]["source"] == "PolitiFact"
    assert sources[0]["title"] == "PolitiFact Check"
    assert sources[0]["url"] == "https://politifact.com/fact-check/test"


# ---------------------------------------------------------------------------
# 14. _validate_sources
# ---------------------------------------------------------------------------


def test_validate_sources_rejects_invented_urls() -> None:
    available = [{"title": "Real", "url": "https://real.com/article", "source": "real.com"}]
    llm_sources = [
        {"title": "Invented", "url": "https://invented-fake.io/article", "source": "fake"},
        {"title": "Real", "url": "https://real.com/article", "source": "real.com"},
    ]
    validated = _validate_sources(llm_sources, available)

    assert len(validated) == 1
    assert validated[0]["url"] == "https://real.com/article"


def test_validate_sources_falls_back_when_all_invented() -> None:
    available = [{"title": "Real", "url": "https://real.com/article", "source": "real.com"}]
    llm_sources = [
        {"title": "Fake", "url": "https://totally-fake.io/nope", "source": "fake"},
    ]
    validated = _validate_sources(llm_sources, available)

    # All LLM sources were invented — fall back to available_sources
    assert validated == available


# ---------------------------------------------------------------------------
# 15. _rule_based_explanation — all verdict branches
# ---------------------------------------------------------------------------


def test_rule_based_explanation_likely_true(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    sources = _build_source_list([supporting_item])
    result = _rule_based_explanation("Claim.", vr, sources)

    assert result.verdict == "LIKELY_TRUE"
    assert "likely true" in result.explanation.lower()
    assert result.llm_used is False
    assert len(result.key_evidence) >= 1


def test_rule_based_explanation_likely_false(contradicting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_FALSE, contradicting=[contradicting_item]
    )
    sources = _build_source_list([contradicting_item])
    result = _rule_based_explanation("Claim.", vr, sources)

    assert result.verdict == "LIKELY_FALSE"
    assert "likely false" in result.explanation.lower()


def test_rule_based_explanation_mixed(
    supporting_item: EvidenceItem, contradicting_item: EvidenceItem
) -> None:
    vr = _make_verification_result(
        VerificationStatus.MIXED,
        supporting=[supporting_item],
        contradicting=[contradicting_item],
    )
    sources = _build_source_list([supporting_item, contradicting_item])
    result = _rule_based_explanation("Claim.", vr, sources)

    assert result.verdict == "MIXED"
    assert "mixed" in result.explanation.lower()


def test_rule_based_explanation_unverified() -> None:
    vr = _make_verification_result(VerificationStatus.UNVERIFIED)
    result = _rule_based_explanation("Claim.", vr, [])

    assert result.verdict == "UNVERIFIED"
    assert "unverified" in result.explanation.lower()


# ---------------------------------------------------------------------------
# 16. llm_used flag
# ---------------------------------------------------------------------------


def test_llm_used_true_when_llm_succeeds(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    provider = StubLLMProvider(response=_valid_llm_json("LIKELY_TRUE"))
    gen = ExplanationGenerator(provider=provider)
    result = gen.generate("Vaccines are effective.", vr)

    assert result.llm_used is True


def test_llm_used_false_when_llm_fails(supporting_item: EvidenceItem) -> None:
    vr = _make_verification_result(
        VerificationStatus.LIKELY_TRUE, supporting=[supporting_item]
    )
    gen = ExplanationGenerator(provider=NoOpLLMProvider())
    result = gen.generate("Vaccines are effective.", vr)

    assert result.llm_used is False


# ---------------------------------------------------------------------------
# NoOpLLMProvider raises LLMError directly
# ---------------------------------------------------------------------------


def test_noop_provider_raises_llm_error() -> None:
    provider = NoOpLLMProvider()
    with pytest.raises(LLMError, match="No LLM API key"):
        provider.complete("sys", "user")


# ---------------------------------------------------------------------------
# create_llm_provider — env-based selection (no real network)
# ---------------------------------------------------------------------------


def test_create_llm_provider_no_keys_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider()
    assert isinstance(provider, NoOpLLMProvider)


def test_create_llm_provider_with_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-openai")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider()
    assert isinstance(provider, OpenAICompatibleProvider)


def test_create_llm_provider_with_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key-groq")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider()
    assert isinstance(provider, OpenAICompatibleProvider)


# ---------------------------------------------------------------------------
# _parse_llm_response edge cases
# ---------------------------------------------------------------------------


def test_parse_llm_response_empty_explanation_uses_fallback() -> None:
    raw = json.dumps({"verdict": "LIKELY_TRUE", "explanation": "", "key_evidence": [], "sources": []})
    available = [{"title": "T", "url": "https://x.com/a", "source": "x.com"}]
    result = _parse_llm_response(raw, "LIKELY_TRUE", available)

    assert result.explanation == "Insufficient evidence to generate an explanation."


def test_parse_llm_response_non_string_key_evidence_is_dropped() -> None:
    """None and empty-string entries are dropped; integers are coerced to str."""
    raw = json.dumps(
        {
            "verdict": "LIKELY_TRUE",
            "explanation": "Good.",
            "key_evidence": [None, "Valid evidence text.", ""],
            "sources": [],
        }
    )
    available = [{"title": "T", "url": "https://x.com/a", "source": "x.com"}]
    result = _parse_llm_response(raw, "LIKELY_TRUE", available)

    # None and empty string are filtered out; "Valid evidence text." survives.
    assert result.key_evidence == ["Valid evidence text."]


def test_parse_llm_response_non_dict_source_is_dropped() -> None:
    available = [{"title": "T", "url": "https://x.com/a", "source": "x.com"}]
    raw = json.dumps(
        {
            "verdict": "LIKELY_TRUE",
            "explanation": "Good.",
            "key_evidence": [],
            "sources": ["not-a-dict", {"url": "https://x.com/a", "title": "T", "source": "x.com"}],
        }
    )
    result = _parse_llm_response(raw, "LIKELY_TRUE", available)

    # "not-a-dict" dropped; the valid dict URL matches available
    assert all(isinstance(s, dict) for s in result.sources)


def test_parse_llm_response_raises_on_invalid_json() -> None:
    available: list[dict] = []
    with pytest.raises(LLMError, match="non-JSON"):
        _parse_llm_response("not json", "UNVERIFIED", available)
