def calculate_overall_score(results):
    """
    Calculate the overall VeriLens veracity score.

    Each claim contributes according to its importance
    and verification confidence.
    """

    if not results:
        return {
            "score": 0,
            "label": "UNCERTAIN",
            "summary": "No claims were available for verification."
        }

    total_weight = 0
    weighted_score = 0

    for result in results:

        importance = result.get(
            "importance",
            "medium"
        ).lower()

        verdict = result.get(
            "verdict",
            "UNCLEAR"
        ).upper()

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        # Importance weights
        if importance == "high":
            weight = 3

        elif importance == "medium":
            weight = 2

        else:
            weight = 1

        # Convert verification result to score
        if verdict == "SUPPORTED":
            claim_score = 100 * confidence

        elif verdict == "CONTRADICTED":
            claim_score = 0

        else:
            # UNCLEAR should be neutral,
            # not treated as completely false.
            claim_score = 50

        weighted_score += (
            claim_score * weight
        )

        total_weight += weight

    if total_weight == 0:
        score = 0

    else:
        score = round(
            weighted_score /
            total_weight
        )

    # Overall label
    if score >= 85:
        label = "HIGHLY RELIABLE"

    elif score >= 70:
        label = "MOSTLY RELIABLE"

    elif score >= 50:
        label = "MIXED / UNCERTAIN"

    elif score >= 30:
        label = "MOSTLY UNRELIABLE"

    else:
        label = "HIGHLY UNRELIABLE"

    summary = (
        f"The article received a "
        f"veracity score of {score}/100."
    )

    return {
        "score": score,
        "label": label,
        "summary": summary
    }