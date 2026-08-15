from src.ai.evidence_retriever import search_wikipedia
from src.ai.verifier import verify_claim


claim = "Chandrayaan-3 landed on the Moon on January 15, 2022"

evidence = search_wikipedia(
    "Chandrayaan-3 landing date"
)


print("\nEVIDENCE:\n")

for item in evidence:
    print(f"Title: {item['title']}")
    print(f"Evidence: {item['snippet']}")
    print()


result = verify_claim(claim, evidence)


print("\nVERIFICATION RESULT:\n")

print("Verdict:", result["verdict"])
print("Confidence:", result["confidence"])
print("Reason:", result["reason"])