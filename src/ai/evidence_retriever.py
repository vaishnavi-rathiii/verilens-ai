import re
import html
import requests


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def clean_snippet(text: str) -> str:
    """Remove Wikipedia search highlighting HTML."""

    text = html.unescape(text)

    text = re.sub(
        r"<span class=\"searchmatch\">",
        "",
        text
    )

    text = text.replace("</span>", "")

    return text


def search_wikipedia(
    query: str,
    limit: int = 5
) -> list[dict]:

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "origin": "*"
    }

    headers = {
        "User-Agent": (
            "VeriLens/1.0 "
            "(AI fact-checking hackathon project)"
        )
    }

    response = requests.get(
        WIKIPEDIA_API,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data.get("query", {}).get("search", []):

        page_id = item.get("pageid")

        title = item.get(
            "title",
            ""
        )

        snippet = clean_snippet(
            item.get("snippet", "")
        )

        results.append({
            "title": title,
            "snippet": snippet,
            "pageid": page_id,
            "url": (
                f"https://en.wikipedia.org/"
                f"?curid={page_id}"
            )
        })

    return results