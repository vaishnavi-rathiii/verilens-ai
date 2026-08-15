from src.ai.evidence_retriever import search_wikipedia
from src.ai.evidence_ranker import rank_evidence
from src.ai.verifier import verify_claim


class VeriLensEngine:

    def __init__(self):
        pass

    def analyze(self, claim: str) -> dict:

        # 1. Retrieve evidence
        evidence = search_wikipedia(
            claim,
            limit=5
        )

        # 2. Rank evidence
        evidence = rank_evidence(
            claim,
            evidence
        )

        # 3. Keep the strongest evidence
        best_evidence = [
            item
            for item in evidence
            if item.get("relevance_score", 0) >= 50
        ][:3]

        # 4. Verify the claim
        verification = verify_claim(
            claim,
            best_evidence
        )

        return {
            "claim": claim,
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "reason": verification["reason"],
            "evidence": best_evidence
        }