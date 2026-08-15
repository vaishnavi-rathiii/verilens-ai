"""Verification verdict logic based on collected evidence.

Step 4 — Evidence Verification
================================
For every EvidenceItem, classify its stance toward the claim as one of:
    SUPPORTING    — the text corroborates the claim
    CONTRADICTING — the text refutes the claim
    NEUTRAL       — the text is related but neither confirms nor denies

Then aggregate those stances into an overall VerificationStatus:
    LIKELY_TRUE   — reliable evidence generally supports the claim
    LIKELY_FALSE  — reliable evidence generally contradicts the claim
    MIXED         — meaningful evidence exists on both sides
    UNVERIFIED    — insufficient reliable evidence to conclude

Design principles
-----------------
* Never invent evidence, URLs, or sources — all decisions are grounded
  exclusively in the EvidenceItem data passed in.
* Does not depend on an LLM for the core classification logic; a
  deterministic keyword-overlap heuristic is used so the module works
  without any API keys.
* Treats retrieved text as untrusted data, not as executable instructions.
* Source authority (via DEFAULT_SOURCE_ENTRIES) is factored into the
  overall verdict but never used to fabricate new evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from rag.evidence import EvidenceStance
from rag.models import EvidenceItem
from rag.sources import DEFAULT_SOURCE_ENTRIES, match_source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — tuned conservatively so that the verifier errs toward
# UNVERIFIED rather than making overconfident claims.
# ---------------------------------------------------------------------------

# Minimum relevance_score for an EvidenceItem to count toward the verdict.
MIN_USEFUL_RELEVANCE: float = 0.10

# Negation-keyword fraction at/above which a text is classified CONTRADICTING.
NEGATION_WEIGHT_THRESHOLD: float = 0.30

# Minimum number of classified (non-NEUTRAL) items required before we will
# produce LIKELY_TRUE or LIKELY_FALSE instead of UNVERIFIED.
MIN_CLASSIFIED_FOR_VERDICT: int = 1

# supporting_weight / (supporting + contradicting) must exceed this for LIKELY_TRUE.
LIKELY_TRUE_THRESHOLD: float = 0.65
# ...and fall below this for LIKELY_FALSE.
LIKELY_FALSE_THRESHOLD: float = 0.35  # = 1 - LIKELY_TRUE_THRESHOLD

# ---------------------------------------------------------------------------
# Negation and contradiction keyword sets (lower-case, unigrams/bigrams)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "false",
        "incorrect",
        "wrong",
        "debunked",
        "misleading",
        "misinformation",
        "disinformation",
        "fake",
        "hoax",
        "myth",
        "untrue",
        "inaccurate",
        "unsubstantiated",
        "refuted",
        "denied",
        "deny",
        "dispute",
        "disputed",
        "disprove",
        "disproved",
        "contradict",
        "contradicts",
        "contradicted",
        "contradicting",
        "baseless",
        "fabricated",
        "exaggerated",
        "lacks evidence",
        "no evidence",
        "found no",
        "there is no",
        "there are no",
        "cannot confirm",
        "could not confirm",
        "has not been",
        "have not been",
    }
)

_SUPPORTING_WORDS: frozenset[str] = frozenset(
    {
        "confirmed",
        "confirm",
        "confirms",
        "verified",
        "verify",
        "verifies",
        "true",
        "accurate",
        "correct",
        "valid",
        "factual",
        "evidence shows",
        "studies show",
        "research shows",
        "data shows",
        "supports",
        "support",
        "supported",
        "consistent",
        "indeed",
        "proven",
        "proves",
        "demonstrate",
        "demonstrates",
        "demonstrated",
        "established",
        "documented",
        "according to",
        "report confirms",
        "found that",
        "shows that",
    }
)


# ---------------------------------------------------------------------------
# Public enums and dataclasses
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """High-level fact-check outcome."""

    LIKELY_TRUE = "LIKELY_TRUE"
    LIKELY_FALSE = "LIKELY_FALSE"
    MIXED = "MIXED"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class ClassifiedEvidence:
    """An EvidenceItem paired with its computed stance."""

    item: EvidenceItem
    stance: EvidenceStance
    # Weight used in the aggregation step (authority + relevance combined).
    weight: float = 1.0


@dataclass
class VerificationResult:
    """Structured verification output for downstream API consumers."""

    status: VerificationStatus
    supporting: list[EvidenceItem] = field(default_factory=list)
    contradicting: list[EvidenceItem] = field(default_factory=list)
    neutral: list[EvidenceItem] = field(default_factory=list)
    summary: str | None = None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class Verifier:
    """Aggregates evidence into a verification status.

    The classification is deterministic and keyword-driven so that it works
    without any external API.  Source authority information from
    ``DEFAULT_SOURCE_ENTRIES`` is used only to up-weight evidence from
    known reliable outlets — it is never used to fabricate new evidence.

    Returns UNVERIFIED when reliable evidence is insufficient.
    """

    def __init__(
        self,
        *,
        min_useful_relevance: float = MIN_USEFUL_RELEVANCE,
        min_classified_for_verdict: int = MIN_CLASSIFIED_FOR_VERDICT,
        likely_true_threshold: float = LIKELY_TRUE_THRESHOLD,
        likely_false_threshold: float = LIKELY_FALSE_THRESHOLD,
    ) -> None:
        if not 0.0 <= min_useful_relevance <= 1.0:
            raise ValueError("min_useful_relevance must be between 0.0 and 1.0.")
        if min_classified_for_verdict < 1:
            raise ValueError("min_classified_for_verdict must be at least 1.")
        if not 0.5 < likely_true_threshold <= 1.0:
            raise ValueError("likely_true_threshold must be in (0.5, 1.0].")
        if not 0.0 <= likely_false_threshold < 0.5:
            raise ValueError("likely_false_threshold must be in [0.0, 0.5).")

        self.min_useful_relevance = min_useful_relevance
        self.min_classified_for_verdict = min_classified_for_verdict
        self.likely_true_threshold = likely_true_threshold
        self.likely_false_threshold = likely_false_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        claim: str,
        evidence: list[EvidenceItem],
    ) -> VerificationResult:
        """Produce a verification result from extracted evidence.

        Args:
            claim: The original claim text (used for logging only; we do not
                   rely on LLM re-evaluation of the claim here).
            evidence: Evidence items produced by the EvidenceExtractor.

        Returns:
            A :class:`VerificationResult` with per-item stances and an
            overall :class:`VerificationStatus`.
        """
        normalized_claim = claim.strip() if claim else ""
        if not normalized_claim:
            logger.warning("Empty claim provided to Verifier; returning UNVERIFIED.")
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                summary="No claim was provided.",
            )

        if not evidence:
            logger.info("No evidence items provided; returning UNVERIFIED.")
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                summary="No evidence was retrieved to evaluate this claim.",
            )

        # --- Classify each item ----------------------------------------
        classified: list[ClassifiedEvidence] = []
        for item in evidence:
            try:
                stance = _classify_stance(item)
                weight = _compute_weight(item)
                classified.append(ClassifiedEvidence(item=item, stance=stance, weight=weight))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Unexpected error classifying evidence item from %s; treating as NEUTRAL.",
                    getattr(item, "url", "<unknown>"),
                )
                classified.append(
                    ClassifiedEvidence(item=item, stance=EvidenceStance.NEUTRAL, weight=0.0)
                )

        # --- Bucket by stance ------------------------------------------
        supporting_items: list[EvidenceItem] = []
        contradicting_items: list[EvidenceItem] = []
        neutral_items: list[EvidenceItem] = []

        for ce in classified:
            if ce.stance is EvidenceStance.SUPPORTING:
                supporting_items.append(ce.item)
            elif ce.stance is EvidenceStance.CONTRADICTING:
                contradicting_items.append(ce.item)
            else:
                neutral_items.append(ce.item)

        # --- Compute weighted totals ------------------------------------
        supporting_weight = sum(
            ce.weight for ce in classified if ce.stance is EvidenceStance.SUPPORTING
        )
        contradicting_weight = sum(
            ce.weight for ce in classified if ce.stance is EvidenceStance.CONTRADICTING
        )

        n_classified = len(supporting_items) + len(contradicting_items)

        # --- Determine overall status ----------------------------------
        status = _determine_status(
            n_classified=n_classified,
            supporting_weight=supporting_weight,
            contradicting_weight=contradicting_weight,
            min_classified=self.min_classified_for_verdict,
            likely_true_threshold=self.likely_true_threshold,
            likely_false_threshold=self.likely_false_threshold,
        )

        summary = _build_summary(
            status=status,
            n_supporting=len(supporting_items),
            n_contradicting=len(contradicting_items),
            n_neutral=len(neutral_items),
        )

        logger.info(
            "Verification complete: status=%s supporting=%d contradicting=%d neutral=%d",
            status.value,
            len(supporting_items),
            len(contradicting_items),
            len(neutral_items),
        )

        return VerificationResult(
            status=status,
            supporting=supporting_items,
            contradicting=contradicting_items,
            neutral=neutral_items,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_keyword_hits(text_lower: str, keyword_set: frozenset[str]) -> int:
    """Count how many keywords from *keyword_set* appear in *text_lower*.

    Supports both unigrams and short multi-word phrases.
    """
    count = 0
    for kw in keyword_set:
        if kw in text_lower:
            count += 1
    return count


def _classify_stance(item: EvidenceItem) -> EvidenceStance:
    """Classify a single EvidenceItem as SUPPORTING, CONTRADICTING, or NEUTRAL.

    The algorithm is purely lexical — no LLM call is made.

    Rules (applied in order):
    1. If the item has no usable text, return NEUTRAL.
    2. Count negation/contradiction keyword hits.
    3. Count supporting keyword hits.
    4. If neither side has hits -> NEUTRAL.
    5. If negation hits dominate (by NEGATION_WEIGHT_THRESHOLD) -> CONTRADICTING.
    6. If supporting hits dominate -> SUPPORTING.
    7. Otherwise -> NEUTRAL (ambiguous).
    """
    text = getattr(item, "text", None) or ""
    text = text.strip()
    if not text:
        return EvidenceStance.NEUTRAL

    text_lower = text.lower()

    neg_hits = _count_keyword_hits(text_lower, _NEGATION_WORDS)
    sup_hits = _count_keyword_hits(text_lower, _SUPPORTING_WORDS)
    total_hits = neg_hits + sup_hits

    if total_hits == 0:
        return EvidenceStance.NEUTRAL

    neg_ratio = neg_hits / total_hits

    if neg_ratio >= NEGATION_WEIGHT_THRESHOLD:
        return EvidenceStance.CONTRADICTING
    if sup_hits > 0:
        return EvidenceStance.SUPPORTING

    return EvidenceStance.NEUTRAL  # fallback


def _compute_weight(item: EvidenceItem) -> float:
    """Assign a weight to an evidence item for aggregation.

    Combines:
    - relevance_score from the EvidenceExtractor (0.0-1.0, default 0.5)
    - authority bonus: +0.5 if the source URL matches a known authoritative entry

    The weight is not normalised — it is used only for relative comparison.
    """
    relevance = item.relevance_score if item.relevance_score is not None else 0.5

    authority_bonus = 0.0
    url = getattr(item, "url", None) or ""
    if url:
        try:
            matched = match_source(url, list(DEFAULT_SOURCE_ENTRIES))
            if matched is not None and matched.authoritative:
                authority_bonus = 0.5
        except ValueError:
            pass  # Malformed URL — treat as unknown source

    return round(relevance + authority_bonus, 4)


def _determine_status(
    *,
    n_classified: int,
    supporting_weight: float,
    contradicting_weight: float,
    min_classified: int,
    likely_true_threshold: float,
    likely_false_threshold: float,
) -> VerificationStatus:
    """Compute the overall VerificationStatus from aggregated weights.

    Args:
        n_classified:         Total number of SUPPORTING + CONTRADICTING items.
        supporting_weight:    Sum of weights for SUPPORTING items.
        contradicting_weight: Sum of weights for CONTRADICTING items.
        min_classified:       Minimum classified items required for a verdict.
        likely_true_threshold:  Supporting fraction above which -> LIKELY_TRUE.
        likely_false_threshold: Supporting fraction below which -> LIKELY_FALSE.

    Returns:
        One of LIKELY_TRUE, LIKELY_FALSE, MIXED, UNVERIFIED.
    """
    if n_classified < min_classified:
        return VerificationStatus.UNVERIFIED

    total_weight = supporting_weight + contradicting_weight
    if total_weight <= 0.0:
        return VerificationStatus.UNVERIFIED

    supporting_fraction = supporting_weight / total_weight

    if supporting_fraction >= likely_true_threshold:
        return VerificationStatus.LIKELY_TRUE
    if supporting_fraction <= likely_false_threshold:
        return VerificationStatus.LIKELY_FALSE

    # Meaningful evidence on both sides.
    return VerificationStatus.MIXED


def _build_summary(
    *,
    status: VerificationStatus,
    n_supporting: int,
    n_contradicting: int,
    n_neutral: int,
) -> str:
    """Return a short, human-readable summary of the verification outcome."""
    total = n_supporting + n_contradicting + n_neutral
    parts: list[str] = [
        f"Analysed {total} evidence item(s): "
        f"{n_supporting} supporting, {n_contradicting} contradicting, {n_neutral} neutral."
    ]

    if status is VerificationStatus.LIKELY_TRUE:
        parts.append("The available evidence generally supports this claim.")
    elif status is VerificationStatus.LIKELY_FALSE:
        parts.append("The available evidence generally contradicts this claim.")
    elif status is VerificationStatus.MIXED:
        parts.append("Evidence is split between supporting and contradicting the claim.")
    else:
        parts.append("There is insufficient reliable evidence to reach a conclusion.")

    return " ".join(parts)

