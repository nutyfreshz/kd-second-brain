from __future__ import annotations

from dataclasses import dataclass
import os

def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    knowledge_source: str
    local_knowledge_dir: str
    allow_anonymous_local: bool
    admin_users: frozenset[str]
    max_concurrent_inference: int
    enable_semantic: bool
    embedding_model: str
    gemini_api_key: str | None
    gemini_model: str
    gemini_free_verified: bool
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_free_verified: bool
    drive_published_folder_id: str | None
    google_service_account_json: str | None
    google_service_account_file: str | None
    drive_external_llm_allow_all: bool

    @classmethod
    def from_env(cls) -> "Settings":
        admins = frozenset(x.strip() for x in os.getenv("ADMIN_USERS", "").split(",") if x.strip())
        return cls(
            knowledge_source=os.getenv("KNOWLEDGE_SOURCE", "local").strip().lower(),
            local_knowledge_dir=os.getenv("LOCAL_KNOWLEDGE_DIR", "knowledge/published"),
            allow_anonymous_local=_bool("ALLOW_ANONYMOUS_LOCAL", False),
            admin_users=admins,
            max_concurrent_inference=max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", "2"))),
            enable_semantic=_bool("ENABLE_SEMANTIC", True),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
            gemini_free_verified=_bool("GEMINI_FREE_VERIFIED", False),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
            openrouter_free_verified=_bool("OPENROUTER_FREE_VERIFIED", False),
            drive_published_folder_id=os.getenv("GOOGLE_DRIVE_PUBLISHED_FOLDER_ID") or None,
            google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or None,
            google_service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or None,
            drive_external_llm_allow_all=_bool("DRIVE_EXTERNAL_LLM_ALLOW_ALL", False),
        )