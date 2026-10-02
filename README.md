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
- Backend validation for citation IDs, verbatim evidence snippets, and critical numbers before an AI answer is shown.
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

`app.yaml` uses the Databricks-provided `DATABRICKS_APP_PORT` and starts Streamlit. Databricks user identity is read from forwarded headers in the Streamlit request context. Keep `ALLOW_ANONYMOUS_LOCAL=false` in Databricks.

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

Core tests do not call external APIs or download embedding models.

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
