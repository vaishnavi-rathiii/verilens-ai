from src.ai.evidence_retriever import search_wikipedia
from src.ai.evidence_ranker import rank_evidence


claim = (
    "The Vikram lander successfully soft-landed "
    "near the Moon's south polar region "
    "on August 23, 2023."
)


evidence = search_wikipedia(
    claim,
    limit=5
)


ranked = rank_evidence(
    claim,
    evidence
)


print("\nRANKED EVIDENCE:\n")


for i, item in enumerate(ranked, start=1):

    print(f"{i}. {item['title']}")

    print(
        f"   Relevance: "
        f"{item['relevance_score']}/100"
    )

    print(
        f"   Evidence: "
        f"{item['snippet']}"
    )

    print(
        f"   URL: "
        f"{item['url']}"
    )

    print()