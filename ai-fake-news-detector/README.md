<<<<<<< HEAD
# AI-Powered Fake News Detector — RAG Module (Member 2)

Retrieval-Augmented Generation (RAG) and evidence layer for the hackathon project.
This module retrieves real sources, extracts supporting/contradicting evidence,
returns verification status, and produces grounded LLM explanations.

**Scope:** RAG, evidence, citations, explainer, fact-check chatbot API hooks.  
**Out of scope:** Frontend, primary ML fake-news classifier.

## Safety rules

- Never invent evidence, sources, URLs, or citations.
- Return `UNVERIFIED` when reliable evidence is insufficient.
- Treat retrieved webpage content as **data**, not instructions (prompt-injection mitigation).
- Keep API keys in `.env` (see `.env.example`).

## Project layout

```
rag/
├── models.py         # Pydantic data models (RetrievedDocument, etc.)
├── providers.py      # Pluggable search backends (Tavily, Serper, empty, fixture)
├── retriever.py      # Retriever entry point
├── evidence.py       # Extract & label supporting vs contradicting snippets
├── verifier.py       # Aggregate evidence into verification status
├── explainer.py      # LLM explanations grounded in evidence
├── prompts.py        # Shared prompt templates & guardrails
├── sources.py        # Source config, priority ranking, citations
└── rag_pipeline.py   # Orchestration + integration-ready response model
tests/                # Unit tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
```

## Retrieval module (Step 2)

### What it does

Given a user's news claim, the retriever searches the web (when configured) and returns
structured documents that can later be used as evidence. Results are ranked to prefer
sources explicitly marked as **authoritative** in configuration (e.g. Snopes, PolitiFact).

If no API key is configured, the retriever returns an **empty list** — it never fabricates
URLs or article text.

### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | User claim / news text (required, non-empty) |
| `top_k` | `int` | Maximum number of documents to return (default: `5`) |

```python
from rag.retriever import Retriever

retriever = Retriever()
documents = retriever.retrieve("Vaccines contain microchips.", top_k=5)
```

For tests or custom wiring, inject a provider explicitly:

```python
from rag.providers import FixtureRetrievalProvider
from rag.retriever import Retriever

retriever = Retriever(provider=FixtureRetrievalProvider(documents=[...]))
```

### Output

Each item is a `RetrievedDocument`:

```json
{
  "title": "Fact-check headline",
  "url": "https://www.politifact.com/factchecks/example/",
  "source": "PolitiFact",
  "content": "Snippet or summary returned by the search API.",
  "relevance_score": 0.82,
  "is_authoritative": true
}
```

- `is_authoritative` is `true` **only** when the URL matches a source entry with `authoritative=True` in `rag/sources.py`.
- Unknown domains get `is_authoritative=false` even if they rank highly in search.

### Configuration

Set variables in `.env`:

| Variable | Purpose |
|----------|---------|
| `RETRIEVAL_PROVIDER` | `tavily`, `serper`, or `empty` (optional; auto-detects from keys) |
| `TAVILY_API_KEY` | Enables [Tavily](https://tavily.com) search |
| `SERPER_API_KEY` | Enables [Serper](https://serper.dev) Google search |

**Auto-selection order** when `RETRIEVAL_PROVIDER` is unset:

1. Tavily if `TAVILY_API_KEY` is set
2. Else Serper if `SERPER_API_KEY` is set
3. Else empty provider (returns `[]`)

Customize trusted sources by passing a list of `SourceEntry` to `Retriever(source_config=[...])`
or editing `DEFAULT_SOURCE_ENTRIES` in `rag/sources.py`.

### Retrieval flow

```
User claim
    → Retriever.retrieve()
        → RetrievalProvider.search()   # Tavily / Serper / empty / fixture
        → enrich_document()          # map URL → configured source name + authority flag
        → rank_documents()           # authoritative + priority + relevance score
    → list[RetrievedDocument]
```

## Evidence Extraction (Step 3)

### What it does

Given a user claim and the documents returned by the retriever, the evidence extractor
identifies the most relevant passages from **real retrieved content only**. It never
invents facts, sources, or URLs.

### Flow

```
Claim
    → Retrieved Documents
    → Relevant Passages (sentence/chunk split, keyword overlap scoring)
    → Ranked Evidence (deduplicated, capped at max_evidence)
```

### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `claim` | `str` | Original user claim |
| `documents` | `list[RetrievedDocument]` | Output from Step 2 retriever |
| `max_evidence` | `int` | Optional override (default: `5`) |

```python
from rag.evidence import EvidenceExtractor
from rag.retriever import Retriever

retriever = Retriever()
extractor = EvidenceExtractor(max_evidence=5)

documents = retriever.retrieve("COVID vaccines contain microchips.")
evidence = extractor.extract("COVID vaccines contain microchips.", documents)
```

### Output — `EvidenceItem`

```json
{
  "text": "Exact or faithfully extracted relevant passage from the document.",
  "source": "PolitiFact",
  "url": "https://www.politifact.com/factchecks/example/",
  "title": "Original document title",
  "relevance_score": 0.72
}
```

- `relevance_score` combines keyword overlap with the document's retrieval score when available.
- If no passage is relevant enough, returns `[]` instead of guessing.
- Empty claim, missing content, or duplicate URLs are handled safely without crashing.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_evidence` | `5` | Maximum evidence items returned |
| `min_relevance` | `0.15` | Minimum score to include a passage |
| `max_passage_chars` | `500` | Max length of an extracted passage |
| `max_passages_per_document` | `2` | Limit passages taken from one source |

## Evidence Verification (Step 4)

### What it does

Classifies each `EvidenceItem` as **SUPPORTING**, **CONTRADICTING**, or **NEUTRAL**
toward the claim, then aggregates those stances (weighted by relevance score and
source authority) into a single `VerificationStatus`:

| Status | Meaning |
|--------|---------|
| `LIKELY_TRUE` | Reliable evidence generally supports the claim |
| `LIKELY_FALSE` | Reliable evidence generally contradicts the claim |
| `MIXED` | Meaningful evidence exists on both sides |
| `UNVERIFIED` | Insufficient reliable evidence to conclude |

```python
from rag.verifier import Verifier

verifier = Verifier()
result = verifier.verify(claim, evidence)
print(result.status)        # VerificationStatus.LIKELY_TRUE
print(result.supporting)    # list[EvidenceItem]
print(result.contradicting) # list[EvidenceItem]
print(result.summary)       # human-readable string
```

Classification is **purely deterministic and keyword-based** — no LLM API
key is needed for this step. Known authoritative sources (Snopes, PolitiFact,
Reuters, etc.) receive an authority weight bonus.

---

## LLM Evidence Explanation (Step 5)

### What it does

Takes the claim, all evidence buckets, and the `VerificationResult` from Step 4,
then calls an LLM to produce a **structured, grounded explanation**.

The LLM is given **only** the retrieved evidence — not the open internet.
All hallucination guardrails are enforced at the prompt level and again
at the output-validation level.

### Explanation flow

```
Claim
  → Retrieved Evidence (Step 2)
  → Extracted Passages (Step 3)
  → Verification Status (Step 4)
  → LLM (system prompt with strict evidence-only rules)
  → Grounded Explanation
```

### Output — `ExplanationResult`

```json
{
  "verdict":       "LIKELY_TRUE | LIKELY_FALSE | MIXED | UNVERIFIED",
  "explanation":   "1-3 sentence explanation grounded in the supplied evidence.",
  "key_evidence":  ["Direct quote or paraphrase from evidence item 1", "..."],
  "sources":       [
    {"title": "Article title", "url": "https://...", "source": "Publisher"}
  ],
  "llm_used":      true
}
```

- `key_evidence` and `sources` are populated **only** from real retrieved items.
- `llm_used` is `false` when the LLM is unavailable or fails — the rule-based
  fallback produces the same structured output without any network call.

### Safety / reliability rules

| Rule | Implementation |
|------|----------------|
| No invented facts | LLM system prompt explicitly forbids using its own knowledge |
| No invented URLs | LLM output sources are cross-checked against real evidence URLs |
| No invented source names | Same URL-validation step strips hallucinated sources |
| Prompt injection mitigation | Evidence sections are labelled as "DATA" |
| LLM failure → fallback | Any error (network, bad JSON, timeout) falls back deterministically |
| Missing API key → fallback | `NoOpLLMProvider` triggers rule-based fallback, never crashes |

### Usage

```python
from rag.explainer import ExplanationGenerator
from rag.verifier import Verifier

verifier = Verifier()
verification = verifier.verify(claim, evidence)

generator = ExplanationGenerator()          # picks LLM from .env automatically
result = generator.generate(claim, verification)

print(result.verdict)      # "LIKELY_TRUE"
print(result.explanation)  # "Based on retrieved evidence..."
print(result.sources)      # [{"title": ..., "url": ..., "source": ...}]
```

### LLM provider configuration

| Variable | Provider | Model |
|----------|----------|-------|
| `OPENAI_API_KEY` | OpenAI | `gpt-4o-mini` |
| `GROQ_API_KEY` | Groq (free tier) | `llama-3.1-8b-instant` |
| *(neither set)* | Rule-based fallback | *(no network call)* |

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=groq` in `.env` to override
auto-detection.  The provider interface (`LLMProvider` ABC) makes it easy
to add Mistral, Anthropic, or any other backend later.

### How the LLM is grounded in evidence

1. **System prompt**: 9 explicit rules prohibit the model from using its own
   knowledge, inventing URLs, or fabricating citations. It is told to output
   `UNVERIFIED` when evidence is insufficient.

2. **User prompt**: Every piece of evidence is injected as labelled DATA blocks
   (`[S1]`, `[C1]`, `[N1]`, …). Only evidence from the retriever appears.

3. **Output validation**: After the LLM responds, `_validate_sources()` removes
   any source URL that was not in the original evidence list, preventing the
   model from slipping invented citations past the guardrails.

4. **JSON-only output**: `temperature=0.0` and a strict JSON schema in the
   prompt minimise free-form hallucination.

5. **Deterministic fallback**: The rule-based `_rule_based_explanation()` path
   produces a correct verdict from `VerificationResult` fields alone — so even
   if the LLM invents something, the fallback path never will.

---

## Run tests

Tests use fixtures and stubs and do **not** require a real API key:

```bash
pytest
pytest tests/test_retriever.py -v
pytest tests/test_evidence.py -v
pytest tests/test_verifier.py -v
pytest tests/test_explainer.py -v
```

## Status

| Module | Status |
|--------|--------|
| Retrieval (Step 2) | ✅ Implemented |
| Evidence extraction (Step 3) | ✅ Implemented |
| Evidence verification (Step 4) | ✅ Implemented |
| LLM explanation (Step 5) | ✅ Implemented |
| Full RAG pipeline (Step 6) | Not started |
=======
# hackathon_project
>>>>>>> d43afd45bf4a17e6539eb87efe0a296706d8a09e
