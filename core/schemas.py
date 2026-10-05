from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

CONTRACT_VERSION = 2

class AnswerStatus(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VALIDATION_FAILED = "validation_failed"
    QUOTA_EXHAUSTED = "quota_exhausted"
    SEARCH_ONLY = "search_only"

@dataclass(frozen=True)
class Identity:
    user_id: str
    username: str
    email: str | None = None
    is_admin: bool = False

@dataclass(frozen=True)
class SourceMeta:
    source_id: str
    doc_id: str
    title: str
    version: str
    status: str
    updated_at: str
    effective_from: str | None = None
    effective_to: str | None = None
    supersedes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    external_llm_allowed: bool = False
    scope: str | None = None
    origin_url: str | None = None
    content_hash: str = ""

@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_id: str
    title: str
    heading: str
    text: str
    ordinal: int
    score: float = 0.0

@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str
    source_id: str
    title: str
    heading: str

@dataclass(frozen=True)
class Claim:
    text: str
    citation_ids: tuple[str, ...]

@dataclass
class ConversationContext:
    selected_source_ids: tuple[str, ...] = ()
    effective_date: str | None = None
    topic: str | None = None
    pending_clarification: tuple[str, ...] = ()

@dataclass
class Message:
    message_id: str
    role: str
    text: str
    created_at: str
    citations: list[Citation] = field(default_factory=list)
    status: str | None = None
    client_message_id: str | None = None
    selected_source_ids: tuple[str, ...] = ()
    clarification_questions: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    snapshot_id: str = ""
    model_used: str | None = None
    limited_mode: bool = False

@dataclass
class Conversation:
    conversation_id: str
    owner_user_id: str
    created_at: str
    context: ConversationContext = field(default_factory=ConversationContext)
    messages: list[Message] = field(default_factory=list)
    processed_client_ids: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ChatRequest:
    conversation_id: str
    client_message_id: str
    text: str
    selected_source_ids: tuple[str, ...] | None = None

@dataclass
class ChatResponse:
    message_id: str
    status: AnswerStatus
    answer_th: str
    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    snapshot_id: str = ""
    model_used: str | None = None
    contract_version: int = CONTRACT_VERSION
    cache_hit: bool = False
    limited_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
