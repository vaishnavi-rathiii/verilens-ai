from src.ai.evidence_retriever import search_wikipedia


claim = (
    "The Vikram lander successfully soft-landed "
    "near the Moon's south polar region "
    "on August 23, 2023."
)


results = search_wikipedia(
    claim,
    limit=5
)


print("\nEVIDENCE SEARCH RESULTS:\n")


for i, result in enumerate(results, start=1):

    print(f"{i}. {result['title']}")

    print(
        f"   Evidence: "
        f"{result['snippet']}"
    )

    print(
        f"   URL: "
        f"{result['url']}"
    )

    print()