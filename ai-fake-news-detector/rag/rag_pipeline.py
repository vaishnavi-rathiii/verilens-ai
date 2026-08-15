"""End-to-end RAG pipeline.

Orchestrates:
1. Retrieval
2. Evidence extraction
3. Evidence verification
4. Grounded explanation
5. Source citation building
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.evidence import EvidenceExtractor
from rag.explainer import Explainer, Explanation
from rag.retriever import Retriever
from rag.sources import SourceCitation, build_citations
from rag.verifier import VerificationResult, Verifier


@dataclass
class RAGResponse:
    """Final response returned by the complete RAG pipeline."""

    claim: str
    verification: VerificationResult
    explanation: Explanation | None
    citations: list[SourceCitation]


class RAGPipeline:
    """End-to-end RAG pipeline entry point."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        verifier: Verifier | None = None,
        explainer: Explainer | None = None,
    ) -> None:

        self.retriever = retriever or Retriever()
        self.evidence_extractor = (
            evidence_extractor or EvidenceExtractor()
        )
        self.verifier = verifier or Verifier()
        self.explainer = explainer or Explainer()

    def run(self, claim: str) -> RAGResponse:
        """Run the complete RAG workflow."""

        # ---------------------------------------------------------
        # STEP 1: Validate input
        # ---------------------------------------------------------

        if not isinstance(claim, str):
            raise TypeError("claim must be a string")

        claim = claim.strip()

        if not claim:
            raise ValueError("claim cannot be empty")

        # ---------------------------------------------------------
        # STEP 2: RETRIEVAL
        # ---------------------------------------------------------

        retrieved_results = self._retrieve(claim)

        # ---------------------------------------------------------
        # STEP 3: EVIDENCE EXTRACTION
        # ---------------------------------------------------------

        evidence = self._extract_evidence(
            claim,
            retrieved_results,
        )

        # ---------------------------------------------------------
        # STEP 4: EVIDENCE VERIFICATION
        # ---------------------------------------------------------

        verification = self._verify(
            claim,
            evidence,
        )

        # ---------------------------------------------------------
        # STEP 5: GROUNDED EXPLANATION
        # ---------------------------------------------------------

        explanation = self._explain(
            claim,
            evidence,
            verification,
        )

        # ---------------------------------------------------------
        # STEP 6: SOURCE CITATIONS
        # ---------------------------------------------------------

        citations = self._build_citations(
            evidence,
            retrieved_results,
        )

        # ---------------------------------------------------------
        # FINAL RESPONSE
        # ---------------------------------------------------------

        return RAGResponse(
            claim=claim,
            verification=verification,
            explanation=explanation,
            citations=citations,
        )

    # =============================================================
    # RETRIEVAL
    # =============================================================

    def _retrieve(self, claim: str) -> Any:
        """Retrieve relevant documents/sources for the claim."""

        if hasattr(self.retriever, "retrieve"):
            return self.retriever.retrieve(claim)

        if hasattr(self.retriever, "search"):
            return self.retriever.search(claim)

        if hasattr(self.retriever, "run"):
            return self.retriever.run(claim)

        raise AttributeError(
            "Retriever must implement one of: "
            "retrieve(), search(), or run()."
        )

    # =============================================================
    # EVIDENCE EXTRACTION
    # =============================================================

    def _extract_evidence(
        self,
        claim: str,
        retrieved_results: Any,
    ) -> Any:
        """Extract useful evidence from retrieved sources."""

        if hasattr(self.evidence_extractor, "extract"):
            try:
                return self.evidence_extractor.extract(
                    claim,
                    retrieved_results,
                )
            except TypeError:
                return self.evidence_extractor.extract(
                    retrieved_results,
                )

        if hasattr(self.evidence_extractor, "extract_evidence"):
            try:
                return self.evidence_extractor.extract_evidence(
                    claim,
                    retrieved_results,
                )
            except TypeError:
                return self.evidence_extractor.extract_evidence(
                    retrieved_results,
                )

        if hasattr(self.evidence_extractor, "run"):
            return self.evidence_extractor.run(
                claim,
                retrieved_results,
            )

        raise AttributeError(
            "EvidenceExtractor must implement "
            "extract(), extract_evidence(), or run()."
        )

    # =============================================================
    # VERIFICATION
    # =============================================================

    def _verify(
        self,
        claim: str,
        evidence: Any,
    ) -> VerificationResult:
        """Classify evidence as supporting, contradicting, or neutral."""

        if hasattr(self.verifier, "verify"):
            try:
                return self.verifier.verify(
                    claim,
                    evidence,
                )
            except TypeError:
                return self.verifier.verify(evidence)

        if hasattr(self.verifier, "check"):
            try:
                return self.verifier.check(
                    claim,
                    evidence,
                )
            except TypeError:
                return self.verifier.check(evidence)

        if hasattr(self.verifier, "run"):
            return self.verifier.run(
                claim,
                evidence,
            )

        raise AttributeError(
            "Verifier must implement verify(), check(), or run()."
        )

    # =============================================================
    # EXPLANATION
    # =============================================================

    def _explain(
        self,
        claim: str,
        evidence: Any,
        verification: VerificationResult,
    ) -> Explanation | None:
        """Generate an explanation grounded in verified evidence."""

        if hasattr(self.explainer, "explain"):
            try:
                return self.explainer.explain(
                    claim,
                    evidence,
                    verification,
                )
            except TypeError:
                try:
                    return self.explainer.explain(
                        claim,
                        evidence,
                    )
                except TypeError:
                    return self.explainer.explain(claim)

        if hasattr(self.explainer, "generate"):
            try:
                return self.explainer.generate(
                    claim,
                    evidence,
                    verification,
                )
            except TypeError:
                try:
                    return self.explainer.generate(
                        claim,
                        evidence,
                    )
                except TypeError:
                    return self.explainer.generate(claim)

        if hasattr(self.explainer, "run"):
            return self.explainer.run(
                claim,
                evidence,
                verification,
            )

        # Explanation is optional in the response.
        return None

    # =============================================================
    # CITATIONS
    # =============================================================

    def _build_citations(
        self,
        evidence: Any,
        retrieved_results: Any,
    ) -> list[SourceCitation]:
        """Build source citations for the final response."""

        # Prefer evidence because it represents the sources
        # actually used during verification.
        try:
            citations = build_citations(evidence)

            if citations is not None:
                return list(citations)

        except (TypeError, AttributeError):
            pass

        # Fall back to retrieved sources if citation building
        # expects the retrieval result.
        try:
            citations = build_citations(retrieved_results)

            if citations is not None:
                return list(citations)

        except (TypeError, AttributeError):
            pass

        return []