from src.ai.claim_extractor import extract_claims
from src.ai.evidence_retriever import search_wikipedia
from src.ai.evidence_ranker import rank_evidence
from src.ai.verifier import verify_claim
from src.ai.scorer import calculate_veracity_score


def analyze_article(article_text: str) -> dict:

    claims = extract_claims(article_text)

    results = []

    for claim_data in claims:

        claim = claim_data["claim"]

        evidence = search_wikipedia(
            claim,
            limit=5
        )

        # Rank evidence before verification
        evidence = rank_evidence(
            claim,
            evidence
        )

        # Only send strongest evidence to verifier
        best_evidence = [
            item
            for item in evidence
            if item["relevance_score"] >= 50
        ][:3]

        verification = verify_claim(
            claim,
            best_evidence
        )

        results.append({
            "claim": claim,
            "importance": claim_data.get(
                "importance",
                "medium"
            ),
            "evidence": best_evidence,
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "reason": verification["reason"]
        })

    overall = calculate_veracity_score(results)

    return {
        "claims": results,
        "overall": overall
    }