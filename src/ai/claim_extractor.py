import json
import re

from src.ai.llm import ask_llm


def extract_claims(article_text: str) -> list[dict]:
    prompt = f"""
You are a professional fact-checking AI.

Extract the most important factual claims from the article below.

A factual claim must be something that can be checked against
reliable external evidence.

Return ONLY a JSON array in this format:

[
  {{
    "claim": "Specific factual statement",
    "importance": "high"
  }}
]

Rules:
- Extract exactly 3 important factual claims.
- Only extract objectively verifiable claims.
- Ignore opinions and speculation.
- Keep claims specific.
- Preserve names, dates, numbers and organizations.
- Do not invent facts.

ARTICLE:
{article_text}
"""

    response = ask_llm(prompt, max_tokens=2000)

    # Find JSON array even if Qwen adds text around it
    match = re.search(r"\[[\s\S]*\]", response)

    if not match:
        print("No JSON array found in LLM response.")
        print("Raw response:")
        print(response)
        return []

    json_text = match.group(0)

    try:
        claims = json.loads(json_text)

    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        print("Extracted JSON:")
        print(json_text)
        return []

    # Validate structure
    valid_claims = []

    for item in claims:

        if not isinstance(item, dict):
            continue

        claim = item.get("claim")
        importance = item.get("importance")

        if claim and importance:
            valid_claims.append({
                "claim": claim,
                "importance": importance
            })

    return valid_claims