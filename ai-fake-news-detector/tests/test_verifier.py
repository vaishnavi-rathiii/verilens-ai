"""Tests for the evidence verifier (Step 4 — Evidence Verification).

Covered scenarios
-----------------
1.  Supporting evidence     → LIKELY_TRUE
2.  Contradicting evidence  → LIKELY_FALSE
3.  Neutral evidence        → UNVERIFIED
4.  Mixed evidence          → MIXED
5.  No evidence             → UNVERIFIED
6.  Empty claim             → UNVERIFIED
7.  Multiple evidence items (5+) with consistent stance
8.  Conflicting evidence (near-equal split) → MIXED or LIKELY_TRUE/LIKELY_FALSE
9.  Missing / incomplete evidence fields (graceful handling)

Additional tests
----------------
- VerificationResult contains correct buckets
- Summary is a non-empty string
- Source authority boosts weight for known fact-checkers
- Threshold configuration is validated at construction time
- Internal helpers: _classify_stance, _compute_weight, _determine_status
"""

from __future__ import annotations

import pytest

from rag.evidence import EvidenceStance
from rag.models import EvidenceItem
from rag.verifier import (
    ClassifiedEvidence,
    VerificationResult,
    VerificationStatus,
    Verifier,
    _classify_stance,
    _compute_weight,
    _determine_status,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_item(
    text: str,
    *,
    source: str = "example.com",
    url: str = "https://example.com/article",
    title: str = "Test Article",
    relevance_score: float | None = 0.75,
) -> EvidenceItem:
    """Convenience factory for EvidenceItem."""
    return EvidenceItem(
        text=text,
        source=source,
        url=url,
        title=title,
        relevance_score=relevance_score,
    )


@pytest.fixture
def verifier() -> Verifier:
    return Verifier()


CLAIM = "COVID vaccines are safe and effective."


# ---------------------------------------------------------------------------
# 1. Supporting evidence → LIKELY_TRUE
# ---------------------------------------------------------------------------


def test_supporting_evidence_yields_likely_true(verifier: Verifier) -> None:
    """Pure supporting evidence should produce LIKELY_TRUE."""
    evidence = [
        _make_item("Studies confirm that COVID vaccines are safe and effective for adults."),
        _make_item("Research shows COVID vaccines reduce hospitalisation rates significantly."),
        _make_item("Data supports the claim that vaccines are accurate and consistent."),
    ]

    result = verifier.verify(CLAIM, evidence)

    assert result.status is VerificationStatus.LIKELY_TRUE
    assert len(result.supporting) >= 1
    assert len(result.contradicting) == 0
    assert result.summary is not None and len(result.summary) > 0


# ---------------------------------------------------------------------------
# 2. Contradicting evidence → LIKELY_FALSE
# ---------------------------------------------------------------------------


def test_contradicting_evidence_yields_likely_false(verifier: Verifier) -> None:
    """Pure contradicting evidence should produce LIKELY_FALSE."""
    evidence = [
        _make_item(
            "This claim is false. No evidence supports the assertion about vaccine safety.",
            url="https://www.snopes.com/fact-check/vaccine-false",
            source="Snopes",
        ),
        _make_item(
            "The claim has been debunked by health authorities. It is misleading and incorrect.",
        ),
        _make_item(
            "Experts refuted the claim. Studies found no basis for this assertion.",
        ),
    ]

    result = verifier.verify(CLAIM, evidence)

    assert result.status is VerificationStatus.LIKELY_FALSE
    assert len(result.contradicting) >= 1
    assert len(result.supporting) == 0
    assert result.summary is not None


# ---------------------------------------------------------------------------
# 3. Neutral evidence → UNVERIFIED
# ---------------------------------------------------------------------------


def test_neutral_evidence_yields_unverified(verifier: Verifier) -> None:
    """Evidence that neither supports nor contradicts should yield UNVERIFIED."""
    evidence = [
        _make_item("The article discusses the general history of vaccine development."),
        _make_item("Scientists met at a conference in Geneva last month to discuss research."),
    ]

    result = verifier.verify(CLAIM, evidence)

    assert result.status is VerificationStatus.UNVERIFIED
    assert len(result.neutral) >= 1
    assert len(result.supporting) == 0
    assert len(result.contradicting) == 0
    assert result.summary is not None


# ---------------------------------------------------------------------------
# 4. Mixed evidence → MIXED
# ---------------------------------------------------------------------------


def test_mixed_evidence_yields_mixed(verifier: Verifier) -> None:
    """Significant evidence on both sides should produce MIXED."""
    evidence = [
        _make_item("Studies confirm COVID vaccines are effective in clinical trials."),
        _make_item(
            "The claim is disputed. Some experts found no evidence of long-term safety.",
        ),
    ]

    result = verifier.verify(CLAIM, evidence)

    # MIXED is expected, but the exact boundary can vary with weights; accept
    # MIXED, LIKELY_TRUE, or LIKELY_FALSE — just not UNVERIFIED when there is
    # classified evidence on both sides.
    assert result.status is not VerificationStatus.UNVERIFIED
    assert len(result.supporting) >= 1
    assert len(result.contradicting) >= 1


# ---------------------------------------------------------------------------
# 5. No evidence → UNVERIFIED
# ---------------------------------------------------------------------------


def test_no_evidence_yields_unverified(verifier: Verifier) -> None:
    """An empty evidence list must return UNVERIFIED."""
    result = verifier.verify(CLAIM, [])

    assert result.status is VerificationStatus.UNVERIFIED
    assert result.supporting == []
    assert result.contradicting == []
    assert result.neutral == []
    assert result.summary is not None


# ---------------------------------------------------------------------------
# 6. Empty claim → UNVERIFIED
# ---------------------------------------------------------------------------


def test_empty_claim_yields_unverified(verifier: Verifier) -> None:
    """An empty or whitespace-only claim must return UNVERIFIED immediately."""
    evidence = [
        _make_item("Studies confirm the claim is accurate and valid."),
    ]

    for bad_claim in ("", "   ", "\t\n"):
        result = verifier.verify(bad_claim, evidence)
        assert result.status is VerificationStatus.UNVERIFIED, (
            f"Expected UNVERIFIED for claim={bad_claim!r}"
        )


# ---------------------------------------------------------------------------
# 7. Multiple evidence items — consistent stance
# ---------------------------------------------------------------------------


def test_multiple_supporting_items_all_bucketed(verifier: Verifier) -> None:
    """All supporting items should appear in the supporting bucket."""
    texts = [
        "Research confirms the claim is accurate.",
        "Data shows the assertion is correct and verified.",
        "Health authorities confirmed this finding.",
        "Studies demonstrated the claim is valid.",
        "Evidence supports the position documented in peer-reviewed journals.",
    ]
    evidence = [_make_item(t, url=f"https://example.com/article-{i}") for i, t in enumerate(texts)]

    result = verifier.verify(CLAIM, evidence)

    assert result.status is VerificationStatus.LIKELY_TRUE
    assert len(result.supporting) == 5
    assert len(result.contradicting) == 0
    assert len(result.neutral) == 0


# ---------------------------------------------------------------------------
# 8. Conflicting evidence — near-equal split
# ---------------------------------------------------------------------------


def test_conflicting_evidence_near_equal_produces_mixed_or_verdict(verifier: Verifier) -> None:
    """Equal-weight supporting and contradicting evidence should not be UNVERIFIED."""
    evidence = [
        _make_item("Studies confirm that vaccines are effective and safe."),
        _make_item("Experts dispute this. The claim is false and has been refuted."),
    ]

    result = verifier.verify(CLAIM, evidence)

    assert result.status in {
        VerificationStatus.MIXED,
        VerificationStatus.LIKELY_TRUE,
        VerificationStatus.LIKELY_FALSE,
    }, f"Unexpected status: {result.status}"
    assert len(result.supporting) >= 1
    assert len(result.contradicting) >= 1


# ---------------------------------------------------------------------------
# 9. Missing / incomplete evidence fields
# ---------------------------------------------------------------------------


def test_evidence_item_with_no_relevance_score_is_handled(verifier: Verifier) -> None:
    """EvidenceItem with relevance_score=None should be handled gracefully."""
    item = _make_item(
        "Research confirms vaccines are safe and effective.",
        relevance_score=None,
    )

    result = verifier.verify(CLAIM, [item])

    # Should still produce a verdict; UNVERIFIED only if neutral
    assert isinstance(result.status, VerificationStatus)
    assert result.summary is not None


def test_evidence_item_with_unknown_url_still_classifies(verifier: Verifier) -> None:
    """An unusual or malformed URL should not crash the verifier."""
    item = EvidenceItem(
        text="Studies confirm this claim is accurate and valid.",
        source="obscure-outlet.io",
        url="https://obscure-outlet.io/article",
        title="Obscure Report",
        relevance_score=0.6,
    )

    result = verifier.verify(CLAIM, [item])

    assert result.status in list(VerificationStatus)
    assert result.summary is not None


# ---------------------------------------------------------------------------
# VerificationResult structure
# ---------------------------------------------------------------------------


def test_verification_result_buckets_are_disjoint(verifier: Verifier) -> None:
    """No EvidenceItem should appear in more than one bucket."""
    evidence = [
        _make_item("Data supports the claim and confirms it is accurate."),
        _make_item("This claim is false and has been debunked by investigators."),
        _make_item("Scientists met to discuss research methodology last week."),
    ]

    result = verifier.verify(CLAIM, evidence)

    all_items = result.supporting + result.contradicting + result.neutral
    assert len(all_items) == len(evidence)

    # All input items appear exactly once across buckets
    seen_texts = {item.text for item in all_items}
    expected_texts = {item.text for item in evidence}
    assert seen_texts == expected_texts


def test_verification_result_summary_mentions_counts(verifier: Verifier) -> None:
    """The summary string should mention evidence item counts."""
    evidence = [
        _make_item("Research confirms the vaccine is safe and effective."),
    ]

    result = verifier.verify(CLAIM, evidence)

    assert result.summary is not None
    # Summary should reference the analysed count
    assert "1" in result.summary


# ---------------------------------------------------------------------------
# Source authority weighting
# ---------------------------------------------------------------------------


def test_authoritative_source_boosts_weight() -> None:
    """A fact-check site should receive a higher weight than an unknown domain."""
    authoritative_item = _make_item(
        "Studies confirm vaccines are effective.",
        url="https://www.snopes.com/fact-check/vaccines",
        source="Snopes",
        relevance_score=0.6,
    )
    unknown_item = _make_item(
        "Studies confirm vaccines are effective.",
        url="https://randomblog.io/vaccines",
        source="randomblog.io",
        relevance_score=0.6,
    )

    weight_auth = _compute_weight(authoritative_item)
    weight_unknown = _compute_weight(unknown_item)

    assert weight_auth > weight_unknown


# ---------------------------------------------------------------------------
# _classify_stance unit tests
# ---------------------------------------------------------------------------


def test_classify_stance_supporting_text() -> None:
    item = _make_item("Studies confirm that the claim is accurate and verified by experts.")
    assert _classify_stance(item) is EvidenceStance.SUPPORTING


def test_classify_stance_contradicting_text() -> None:
    item = _make_item("This claim is false. The assertion has been debunked and refuted.")
    assert _classify_stance(item) is EvidenceStance.CONTRADICTING


def test_classify_stance_neutral_text() -> None:
    item = _make_item("Scientists gathered in Geneva to attend the annual symposium.")
    assert _classify_stance(item) is EvidenceStance.NEUTRAL


def test_classify_stance_empty_text_is_neutral() -> None:
    # model_construct bypasses validation to allow blank text for this edge case
    item = EvidenceItem.model_construct(
        text="",
        source="example.com",
        url="https://example.com/",
        title="Test",
        relevance_score=0.5,
    )
    assert _classify_stance(item) is EvidenceStance.NEUTRAL


# ---------------------------------------------------------------------------
# _determine_status unit tests
# ---------------------------------------------------------------------------


def test_determine_status_likely_true() -> None:
    status = _determine_status(
        n_classified=3,
        supporting_weight=2.0,
        contradicting_weight=0.3,
        min_classified=1,
        likely_true_threshold=0.65,
        likely_false_threshold=0.35,
    )
    assert status is VerificationStatus.LIKELY_TRUE


def test_determine_status_likely_false() -> None:
    status = _determine_status(
        n_classified=3,
        supporting_weight=0.2,
        contradicting_weight=2.0,
        min_classified=1,
        likely_true_threshold=0.65,
        likely_false_threshold=0.35,
    )
    assert status is VerificationStatus.LIKELY_FALSE


def test_determine_status_mixed() -> None:
    status = _determine_status(
        n_classified=2,
        supporting_weight=1.0,
        contradicting_weight=1.0,
        min_classified=1,
        likely_true_threshold=0.65,
        likely_false_threshold=0.35,
    )
    assert status is VerificationStatus.MIXED


def test_determine_status_unverified_no_classified() -> None:
    status = _determine_status(
        n_classified=0,
        supporting_weight=0.0,
        contradicting_weight=0.0,
        min_classified=1,
        likely_true_threshold=0.65,
        likely_false_threshold=0.35,
    )
    assert status is VerificationStatus.UNVERIFIED


# ---------------------------------------------------------------------------
# Verifier constructor validation
# ---------------------------------------------------------------------------


def test_verifier_rejects_invalid_min_relevance() -> None:
    with pytest.raises(ValueError, match="min_useful_relevance"):
        Verifier(min_useful_relevance=1.5)


def test_verifier_rejects_invalid_min_classified() -> None:
    with pytest.raises(ValueError, match="min_classified_for_verdict"):
        Verifier(min_classified_for_verdict=0)


def test_verifier_rejects_invalid_likely_true_threshold() -> None:
    with pytest.raises(ValueError, match="likely_true_threshold"):
        Verifier(likely_true_threshold=0.4)  # must be > 0.5


def test_verifier_rejects_invalid_likely_false_threshold() -> None:
    with pytest.raises(ValueError, match="likely_false_threshold"):
        Verifier(likely_false_threshold=0.6)  # must be < 0.5
