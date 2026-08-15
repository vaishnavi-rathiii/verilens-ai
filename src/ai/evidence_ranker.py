import re


STOPWORDS = {
    "the", "a", "an", "is", "was", "were",
    "on", "in", "at", "to", "of", "and",
    "or", "for", "with", "near", "from",
    "by", "this", "that", "has", "have",
    "had", "its", "it", "as", "be"
}


def tokenize(text: str) -> set[str]:

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 2
    }


def extract_numbers(text: str) -> set[str]:

    return set(
        re.findall(
            r"\b\d+(?:[-/]\d+)*\b",
            text
        )
    )


def extract_key_phrases(text: str) -> set[str]:

    text = text.lower()

    phrases = [
        "soft landing",
        "south polar",
        "south pole",
        "lunar orbit",
        "moon landing",
        "launched",
        "landed",
        "successfully landed"
    ]

    return {
        phrase
        for phrase in phrases
        if phrase in text
    }


def calculate_score(
    claim: str,
    source_text: str
) -> int:

    claim_words = tokenize(claim)
    source_words = tokenize(source_text)

    if not claim_words:
        return 0

    # Normal word overlap
    common_words = (
        claim_words.intersection(source_words)
    )

    word_score = (
        len(common_words) /
        len(claim_words)
    )

    # Numbers are highly important in fact checking
    claim_numbers = extract_numbers(claim)
    source_numbers = extract_numbers(source_text)

    number_score = 0

    if claim_numbers:

        matching_numbers = (
            claim_numbers.intersection(
                source_numbers
            )
        )

        number_score = (
            len(matching_numbers) /
            len(claim_numbers)
        )

    # Important phrases
    claim_phrases = extract_key_phrases(claim)
    source_phrases = extract_key_phrases(source_text)

    phrase_score = 0

    if claim_phrases:

        matching_phrases = (
            claim_phrases.intersection(
                source_phrases
            )
        )

        phrase_score = (
            len(matching_phrases) /
            len(claim_phrases)
        )

    # Weighted score
    score = (
        word_score * 50
        + number_score * 25
        + phrase_score * 25
    )

    return round(
        min(score, 100)
    )


def rank_evidence(
    claim: str,
    evidence: list[dict]
) -> list[dict]:

    ranked = []

    for item in evidence:

        source_text = (
            item.get("title", "")
            + " "
            + item.get("snippet", "")
        )

        score = calculate_score(
            claim,
            source_text
        )

        ranked.append({
            **item,
            "relevance_score": score
        })

    ranked.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return ranked