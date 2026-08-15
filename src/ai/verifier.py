import json

from src.ai.llm import ask_llm


def verify_claim(claim: str, evidence: list[dict]) -> dict:

    evidence_text = "\n\n".join(
        [
            f"Source: {item.get('title', '')}\n"
            f"Evidence: {item.get('snippet', '')}"
            for item in evidence
        ]
    )

    prompt = f"""
You are a professional fact-checking AI.

Your task is to determine whether the CLAIM is supported
or contradicted by the provided evidence.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

Possible verdicts:

SUPPORTED
- The evidence directly supports the claim.

CONTRADICTED
- The evidence directly conflicts with the claim.

UNCLEAR
- The evidence is insufficient to determine whether the claim
  is true or false.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

"verdict"
"confidence"
"reason"

Rules:

1. verdict must be exactly one of:
   SUPPORTED
   CONTRADICTED
   UNCLEAR

2. confidence must be a number between 0 and 1.

3. The reason must explain WHY the evidence supports,
   contradicts, or fails to establish the claim.

4. Mention specific dates, names, numbers, or facts from
   the evidence when they are relevant.

5. Never use a generic reason such as:
   "The evidence supports the claim."

6. Never invent facts that are not present in the evidence.

7. Keep the reason to 1-2 sentences.

Example output:

{{
  "verdict": "SUPPORTED",
  "confidence": 0.90,
  "reason": "The Chandrayaan-3 source states that the mission was launched on July 14, 2023, matching the launch date in the claim."
}}
"""

    raw_response = ask_llm(
        prompt,
        max_tokens=1000
    )

    print("\n===== VERIFIER RAW RESPONSE =====")
    print(raw_response)
    print("=================================\n")

    try:

        result = json.loads(raw_response)

        verdict = result.get("verdict", "UNCLEAR")
        confidence = float(result.get("confidence", 0))
        reason = result.get(
            "reason",
            "The evidence was insufficient to determine the claim."
        )

        if verdict not in {
            "SUPPORTED",
            "CONTRADICTED",
            "UNCLEAR"
        }:
            verdict = "UNCLEAR"

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason
        }

    except (json.JSONDecodeError, TypeError, ValueError):

        return {
            "verdict": "UNCLEAR",
            "confidence": 0.0,
            "reason": "The AI response could not be parsed."
        }