from rag_engine import FakeNewsRAG
from src.ai.verifier import verify_claim


class VeriLensEngine:

    def __init__(self):
        self.rag = FakeNewsRAG()

    def analyze(self, claim: str) -> dict:

        # Member 2: retrieve evidence
        rag_evidence = self.rag.get_evidence(claim)

        # Convert Member 2 format → Member 1 format
        verifier_evidence = []

        for item in rag_evidence:
            verifier_evidence.append({
                "title": item.get("source", "Unknown"),
                "snippet": item.get("text", "")
            })

        # Member 1: verify claim
        verification = verify_claim(
            claim,
            verifier_evidence
        )

        return {
            "claim": claim,
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "reason": verification["reason"],
            "evidence": rag_evidence
        }