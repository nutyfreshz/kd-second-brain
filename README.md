# KD Second Brain

Conversational RAG prototype for a small internal team, designed for Databricks Apps.

## Current implementation

- Streamlit chat UI with source selection and citation viewer.
- RAG core is framework-independent. `core/` does not import Streamlit.
- Atomic knowledge snapshots from either reviewed local Markdown or Google Drive.
- Thai-friendly lexical BM25 retrieval plus optional local multilingual semantic retrieval with rank fusion.
- Conversation ownership, source scoping, exact per-session cache, and duplicate `client_message_id` protection.
- Gemini adapter as primary, OpenRouter free-model adapter as availability fallback.
- Fail-closed provider behavior. Without a validated free LLM route, the app stays usable in evidence-only search mode.
- Backend validation for nonempty claims, citation IDs, verbatim evidence, and supported numeric units before an AI answer is shown. Semantic entailment still requires live QA.
- Databricks forwarded-header identity adapter. Local anonymous mode is opt-in only.

## Security boundary

This repository is public. **Do not commit AIS/internal Markdown, API keys, Google credentials, Databricks tokens, exported chats, or provider responses containing internal content.**

Use one of these at runtime:

1. Google Drive `published` folder through a read-only Google identity.
2. A reviewed local bundle mounted/deployed into `knowledge/published/`.

The local knowledge directory is intentionally empty in Git.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
```

Set `ALLOW_ANONYMOUS_LOCAL=true` for local-only development, then:

```bash
streamlit run ui/streamlit_app.py
```

If no provider key is configured, the app runs in `search-only` mode. This is intentional.

## Provider setup

Provider transport is implemented but not considered production-enabled until credentials, zero-cost eligibility, model ID, quota and Thai QA are verified.

- Primary: Gemini direct API.
- Fallback: OpenRouter only for the explicitly configured free model.
- Having a key is not enough. `GEMINI_FREE_VERIFIED=true` / `OPENROUTER_FREE_VERIFIED=true` must be set only after live zero-cost eligibility and the exact allowlisted model are verified.
- A model returning `not_found`, `clarify`, or `conflict` does not trigger fallback.
- Safety blocks do not trigger fallback.

Secrets are read from environment variables only. See `.env.example`.

## Google Drive setup

Set:

- `KNOWLEDGE_SOURCE=drive`
- `GOOGLE_DRIVE_PUBLISHED_FOLDER_ID=<folder-id>`
- `GOOGLE_SERVICE_ACCOUNT_JSON=<service-account-json>` or `GOOGLE_SERVICE_ACCOUNT_FILE=<path>`

The adapter is read-only and paginates through every child in the configured folder. The application builds a complete new snapshot before swapping it active. If sync fails, the previous snapshot stays active.

## Databricks Apps

`app.yaml` starts Streamlit. Port binding and connectivity must be verified in the actual Databricks Apps runtime. Databricks user identity is read from forwarded headers in the Streamlit request context. Keep `ALLOW_ANONYMOUS_LOCAL=false` in Databricks.

External connectivity still needs to be verified from the actual app runtime for Google Drive, Gemini/OpenRouter, and any model-download host used by local embeddings.

## Architecture

```text
Streamlit UI
    ↓
ChatService
    ├── ConversationStore
    ├── KnowledgeManager
    │     ├── Markdown parser
    │     └── Hybrid Retriever
    ├── Answer Policy
    ├── Provider Chain
    │     ├── Gemini
    │     └── OpenRouter
    └── Validators
```

The UI only renders typed service responses. Provider output is normalized before UI rendering. This allows a future FastAPI/custom webchat adapter to reuse the same core.

## Tests

Core tests do not call external APIs or download embedding models. The UI integration test uses Streamlit AppTest. NumPy is used by the fake semantic encoder test (also installed by sentence-transformers).

```bash
python -m unittest discover -s tests -v
```

## Deliberately deferred

- Real provider credential walkthrough and live inference QA.
- Google Drive folder creation/publishing workflow.
- Production admin group mapping.
- Persistent chat history.
- HTTP/FastAPI adapter and custom React webchat.
- Live 10-user concurrency/load acceptance.
- Pinning exact dependency versions after Databricks runtime compatibility is tested.


## RAG quality audit (2026-10-05)

See [audit, measured results, trade-offs and live acceptance gate](docs/RAG_QUALITY_AUDIT.md).

- Source filtering happens before candidate limits in both retrieval paths.
- Sparse lexical search uses postings; long semantic passages use token windows.
- Answer text is rendered from validated claims with numbered evidence references.
- Citation IDs include document content hashes; conversation citations retain their original text after sync.
- Complete turns are serialized per user for concurrent duplicate protection; answer cache is bounded and context/date/prompt-aware.
- An unchanged sync reuses the active index.
- Streamlit exposes all-source and explicit-source modes with an empty-selection guard.

Contract v2: omit `selected_source_ids` / use `None` to search all authoritative documents; an empty tuple/list means no sources. This is a deliberate change from v1, where an empty selection meant all sources.

```bash
python benchmarks/retrieval_quality.py
# Historical comparison, if that revision is available locally:
python benchmarks/retrieval_quality.py --baseline 82f1fc4
```

The checked-in `app.yaml` currently opts into **paid OpenRouter UAT**. It is not a zero-cost deployment configuration. This audit preserves that pre-existing deployment choice; free fallback routes are now independently gated against paid models. Do not assume a model ID or quota is live-verified merely because its adapter is present.

`.env.example` is a configuration reference. This app currently reads process environment variables; copying it to `.env` alone does not load those variables. Export them in your shell or configure them through Databricks Apps resources.
