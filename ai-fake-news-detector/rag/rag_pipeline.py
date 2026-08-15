"""End-to-end RAG pipeline.

Orchestrates:
1. Retrieval
2. Evidence extraction
3. Evidence verification
4. Grounded explanation
5. Source citation building
"""

from __future__ import annotations
# from typing import Any

from dataclasses import dataclass

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
        # STEP 2: Retrieval
        # ---------------------------------------------------------

        retrieved_results = self.retriever.retrieve(claim)

        # ---------------------------------------------------------
        # STEP 3: Evidence extraction
        # ---------------------------------------------------------

        evidence = self.evidence_extractor.extract(
            claim,
            retrieved_results,
        )

        # ---------------------------------------------------------
        # STEP 4: Verification
        # ---------------------------------------------------------

        verification = self.verifier.verify(
            claim,
            evidence,
        )

        # ---------------------------------------------------------
        # STEP 5: Grounded explanation
        # ---------------------------------------------------------

        explanation = self.explainer.explain(
            claim,
            verification,
        )

        # ---------------------------------------------------------
        # STEP 6: Source citations
        # ---------------------------------------------------------

        citations = build_citations(evidence)

        # ---------------------------------------------------------
        # FINAL RESPONSE
        # ---------------------------------------------------------

        return RAGResponse(
            claim=claim,
            verification=verification,
            explanation=explanation,
            citations=citations,
        )