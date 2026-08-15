from src.ai.scorer import calculate_veracity_score


claims = [
    {
        "claim": "Claim 1",
        "importance": "high",
        "verdict": "SUPPORTED",
        "confidence": 0.95
    },
    {
        "claim": "Claim 2",
        "importance": "high",
        "verdict": "SUPPORTED",
        "confidence": 0.90
    },
    {
        "claim": "Claim 3",
        "importance": "high",
        "verdict": "CONTRADICTED",
        "confidence": 0.95
    }
]


result = calculate_veracity_score(claims)

print("\nVERILENS SCORE")
print("-----------------------")
print("Score:", result["score"])
print("Label:", result["label"])
print("Summary:", result["summary"])