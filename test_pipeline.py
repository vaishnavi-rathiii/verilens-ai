from src.ai.claim_extractor import extract_claims
from src.ai.evidence_retriever import search_wikipedia
from src.ai.evidence_ranker import rank_evidence
from src.ai.verifier import verify_claim
from src.ai.scorer import calculate_overall_score


ARTICLE = """
The Indian Space Research Organisation launched Chandrayaan-3
on July 14, 2023, from the Satish Dhawan Space Centre.

The Vikram lander successfully soft-landed near the Moon's
south polar region on August 23, 2023.

Social media posts claiming that Chandrayaan-3 landed in
January 2022 are false because official mission records
show the mission launched in July 2023 and landed in August 2023.
"""


print("\nExtracting claims...\n")

claims = extract_claims(ARTICLE)

results = []


for claim_data in claims:

    claim = claim_data["claim"]
    importance = claim_data.get("importance", "medium")

    print("\n" + "=" * 60)
    print("CLAIM")
    print("=" * 60)

    print(claim)

    print("\nSearching for evidence...")

    evidence = search_wikipedia(
        claim,
        limit=5
    )

    ranked_evidence = rank_evidence(
        claim,
        evidence
    )

    # Only send the best 3 sources to the verifier.
    selected_evidence = ranked_evidence[:3]

    print("\nTop Evidence:")

    for source in selected_evidence:

        print(
            f"  • {source['title']} "
            f"({source['relevance_score']}/100)"
        )

    print("\nVerifying claim...")

    verification = verify_claim(
        claim,
        selected_evidence
    )

    results.append({
        "claim": claim,
        "importance": importance,
        "verdict": verification.get(
            "verdict",
            "UNCLEAR"
        ),
        "confidence": verification.get(
            "confidence",
            0
        ),
        "reason": verification.get(
            "reason",
            ""
        ),
        "evidence": selected_evidence
    })


# --------------------------------------------------
# OVERALL SCORE
# --------------------------------------------------

overall = calculate_overall_score(results)


print("\n")
print("=" * 60)
print("                 OVERALL RESULT")
print("=" * 60)

print(
    f"\nVeriLens Score: "
    f"{overall['score']}/100"
)

print(
    f"Label: "
    f"{overall['label']}"
)

print(
    f"Summary: "
    f"{overall['summary']}"
)


# --------------------------------------------------
# CLAIM RESULTS
# --------------------------------------------------

print("\n")
print("=" * 60)
print("                 CLAIM RESULTS")
print("=" * 60)


for i, item in enumerate(results, start=1):

    print("\n" + "-" * 60)

    print(f"CLAIM {i}")

    print("-" * 60)

    print(
        f"Claim: {item['claim']}"
    )

    print(
        f"Importance: {item['importance']}"
    )

    print(
        f"Verdict: {item['verdict']}"
    )

    print(
        f"Confidence: "
        f"{item['confidence'] * 100:.0f}%"
    )

    print(
        f"Reason: {item['reason']}"
    )

    print("\nEvidence:")

    for source in item["evidence"]:

        print(
            f"  • {source['title']} "
            f"({source['relevance_score']}/100)"
        )

        print(
            f"    {source['url']}"
        )