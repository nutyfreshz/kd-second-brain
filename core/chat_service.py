from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import uuid

from adapters.conversation_store import InMemoryConversationStore
from adapters.providers.base import LLMProvider, ProviderError
from core.answer_policy import (
    build_evidence_prompt,
    evidence_only_answer,
    parse_model_json,
)
from core.conversation import retrieval_query, resolve_effective_date
from core.kb_sync import KnowledgeManager, resolve_authoritative_sources
from core.schemas import (
    AnswerStatus,
    ChatRequest,
    ChatResponse,
    Citation,
    Claim,
    Identity,
    Message,
    SourceMeta,
    utc_now_iso,
)
from core.validators import validate_answer


class ChatService:
    def __init__(
        self,
        *,
        knowledge: KnowledgeManager,
        store: InMemoryConversationStore,
        primary: LLMProvider | None = None,
        fallback: LLMProvider | None = None,
        max_concurrent_inference: int = 2,
        system_prompt_path: str = "prompts/SYSTEM_PROMPT_TH.md",
        allow_external_llm_for_all_evidence: bool = False,
    ):
        self.knowledge = knowledge
        self.store = store
        self.primary = primary
        self.fallback = fallback
        self._global_gate = threading.BoundedSemaphore(max_concurrent_inference)
        self._user_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, ChatResponse] = {}
        self.system_prompt_path = Path(system_prompt_path)
        self.allow_external_llm_for_all_evidence = allow_external_llm_for_all_evidence

    def create_conversation(self, identity: Identity) -> str:
        return self.store.create(identity).conversation_id

    def get_conversation(self, identity: Identity, conversation_id: str):
        return self.store.get(conversation_id, identity)

    def list_sources(self, identity: Identity) -> list[SourceMeta]:
        _ = identity
        snapshot = self.knowledge.snapshot
        return list(snapshot.sources) if snapshot else []

    def get_citation(
        self, identity: Identity, conversation_id: str, chunk_id: str
    ) -> Citation:
        self.store.get(conversation_id, identity)
        snapshot = self.knowledge.snapshot
        if not snapshot:
            raise KeyError(chunk_id)
        chunk = next((c for c in snapshot.chunks if c.chunk_id == chunk_id), None)
        if chunk is None:
            raise KeyError(chunk_id)
        return Citation(
            chunk_id=chunk.chunk_id,
            quote=chunk.text,
            source_id=chunk.source_id,
            title=chunk.title,
            heading=chunk.heading,
        )

    def sync_knowledge(self, identity: Identity):
        if not identity.is_admin:
            raise PermissionError("admin_required")
        snapshot = self.knowledge.sync()
        self._cache.clear()
        return snapshot

    def send_message(self, identity: Identity, request: ChatRequest) -> ChatResponse:
        conversation = self.store.get(request.conversation_id, identity)
        duplicate = self.store.response_for_client_id(
            request.conversation_id, identity, request.client_message_id
        )
        if duplicate and duplicate.role == "assistant":
            return self._response_from_message(duplicate, cache_hit=True)

        snapshot = self.knowledge.snapshot
        if snapshot is None:
            snapshot = self.knowledge.sync()

        if not snapshot.sources:
            response = ChatResponse(
                message_id=str(uuid.uuid4()),
                status=AnswerStatus.NOT_FOUND,
                answer_th="คลังความรู้ยังไม่มีเอกสาร published ที่พร้อมใช้งาน",
                snapshot_id=snapshot.snapshot_id,
                limited_mode=True,
            )
            return self._save_response(identity, request, response)

        query, context = retrieval_query(conversation, request.text)
        as_of = resolve_effective_date(
            request.text, conversation.context.effective_date
        )
        conversation.context.effective_date = as_of.isoformat()

        authoritative_ids, conflicts = resolve_authoritative_sources(
            snapshot.sources, as_of
        )
        if conflicts:
            response = ChatResponse(
                message_id=str(uuid.uuid4()),
                status=AnswerStatus.CONFLICT,
                answer_th=(
                    "พบเอกสารหลายเวอร์ชันที่ยังระบุ authority ไม่ได้สำหรับช่วงวันที่นี้: "
                    + ", ".join(conflicts)
                ),
                snapshot_id=snapshot.snapshot_id,
                limited_mode=snapshot.retriever.limited_mode,
            )
            return self._save_response(identity, request, response)

        selected = (
            set(request.selected_source_ids)
            if request.selected_source_ids
            else set(authoritative_ids)
        )
        selected &= authoritative_ids

        user_message = Message(
            message_id=str(uuid.uuid4()),
            role="user",
            text=request.text,
            created_at=utc_now_iso(),
            client_message_id=None,
            selected_source_ids=tuple(sorted(selected)),
        )
        self.store.add_message(request.conversation_id, identity, user_message)

        evidence = snapshot.retriever.search(
            query,
            allowed_source_ids=selected,
            candidate_k=12,
            evidence_k=6,
        )
        if not evidence:
            response = ChatResponse(
                message_id=str(uuid.uuid4()),
                status=AnswerStatus.NOT_FOUND,
                answer_th="ไม่พบข้อมูลที่ตอบคำถามนี้ในแหล่งข้อมูลที่เลือก",
                snapshot_id=snapshot.snapshot_id,
                limited_mode=snapshot.retriever.limited_mode,
            )
            return self._save_response(identity, request, response)

        cache_key = self._cache_key(
            conversation_id=request.conversation_id,
            question=query,
            selected=selected,
            snapshot_id=snapshot.snapshot_id,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached = replace(
                cached,
                message_id=str(uuid.uuid4()),
                cache_hit=True,
            )
            return self._save_response(identity, request, cached)

        evidence_source_ids = {c.source_id for c in evidence}
        meta_by_id = {s.source_id: s for s in snapshot.sources}
        external_allowed = self.allow_external_llm_for_all_evidence or all(
            meta_by_id[sid].external_llm_allowed
            for sid in evidence_source_ids
            if sid in meta_by_id
        )

        provider = self._first_enabled_provider()
        if provider is None or not external_allowed:
            response = self._search_only_response(snapshot, evidence)
            if provider is None:
                response.status = AnswerStatus.PROVIDER_UNAVAILABLE
                response.answer_th = (
                    "ระบบค้นหลักฐานได้ แต่ AI provider ยังไม่พร้อมใช้งาน "
                    "จึงยังไม่สามารถสังเคราะห์คำตอบแบบสนทนาได้"
                )
            else:
                response.answer_th = (
                    "พบหลักฐานแล้ว แต่เอกสารชุดนี้ยังไม่ได้อนุญาตให้ส่งไป external LLM "
                    "จึงแสดงเฉพาะหลักฐาน"
                )
            self._cache[cache_key] = response
            return self._save_response(identity, request, response)

        system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        user_prompt = build_evidence_prompt(request.text, evidence, context)

        with self._get_user_lock(identity.user_id), self._global_gate:
            try:
                result = provider.generate(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )
            except ProviderError as exc:
                if exc.safety_block:
                    response = self._search_only_response(snapshot, evidence)
                    response.status = AnswerStatus.PROVIDER_UNAVAILABLE
                    response.answer_th = (
                        "AI ไม่พร้อมสำหรับคำถามนี้ แต่ยังเปิดหลักฐานที่ค้นพบได้"
                    )
                    return self._save_response(identity, request, response)
                if exc.quota_exhausted:
                    response = self._search_only_response(snapshot, evidence)
                    response.status = AnswerStatus.QUOTA_EXHAUSTED
                    response.answer_th = (
                        "โควตา AI ฟรีไม่พร้อมใช้งานขณะนี้ แต่ยังค้นหลักฐานได้"
                    )
                    return self._save_response(identity, request, response)
                fallback = self._fallback_if_allowed(provider, exc)
                if fallback is None:
                    response = self._search_only_response(snapshot, evidence)
                    response.status = AnswerStatus.PROVIDER_UNAVAILABLE
                    response.answer_th = (
                        "ระบบค้นหลักฐานได้ แต่ AI provider ตอบคำขอไม่สำเร็จ "
                        "จึงยังไม่สามารถสังเคราะห์คำตอบแบบสนทนาได้"
                    )
                    return self._save_response(identity, request, response)
                try:
                    result = fallback.generate(
                        system_prompt=system_prompt, user_prompt=user_prompt
                    )
                except ProviderError:
                    response = self._search_only_response(snapshot, evidence)
                    response.status = AnswerStatus.PROVIDER_UNAVAILABLE
                    response.answer_th = (
                        "ระบบค้นหลักฐานได้ แต่ AI providers ไม่พร้อมใช้งาน "
                        "จึงยังไม่สามารถสังเคราะห์คำตอบแบบสนทนาได้"
                    )
                    return self._save_response(identity, request, response)

        try:
            parsed = parse_model_json(result.text, evidence)
        except Exception:
            response = self._search_only_response(snapshot, evidence)
            response.status = AnswerStatus.VALIDATION_FAILED
            response.answer_th = (
                "คำตอบ AI ไม่ผ่านรูปแบบตรวจสอบ จึงแสดงเฉพาะหลักฐานที่ค้นพบ"
            )
            return self._save_response(identity, request, response)

        if parsed["status"] != AnswerStatus.ANSWER:
            response = ChatResponse(
                message_id=str(uuid.uuid4()),
                status=parsed["status"],
                answer_th=parsed["answer_th"],
                claims=parsed["claims"],
                citations=parsed["citations"],
                clarification_questions=parsed["clarification_questions"],
                snapshot_id=snapshot.snapshot_id,
                model_used=result.model,
                limited_mode=snapshot.retriever.limited_mode,
            )
            self._cache[cache_key] = response
            return self._save_response(identity, request, response)

        validation = validate_answer(
            claims=parsed["claims"],
            citations=parsed["citations"],
            evidence=evidence,
            user_text=request.text,
        )
        if not validation.valid:
            response = self._search_only_response(snapshot, evidence)
            response.status = AnswerStatus.VALIDATION_FAILED
            response.answer_th = (
                "คำตอบ AI ไม่ผ่าน evidence validation จึงแสดงเฉพาะหลักฐานที่ค้นพบ"
            )
            return self._save_response(identity, request, response)

        response = ChatResponse(
            message_id=str(uuid.uuid4()),
            status=AnswerStatus.ANSWER,
            answer_th=parsed["answer_th"],
            claims=parsed["claims"],
            citations=parsed["citations"],
            clarification_questions=parsed["clarification_questions"],
            snapshot_id=snapshot.snapshot_id,
            model_used=result.model,
            limited_mode=snapshot.retriever.limited_mode,
        )
        self._cache[cache_key] = response
        return self._save_response(identity, request, response)

    def _search_only_response(self, snapshot, evidence) -> ChatResponse:
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                quote=c.text,
                source_id=c.source_id,
                title=c.title,
                heading=c.heading,
            )
            for c in evidence
        ]
        return ChatResponse(
            message_id=str(uuid.uuid4()),
            status=AnswerStatus.SEARCH_ONLY,
            answer_th=evidence_only_answer(evidence),
            citations=citations,
            snapshot_id=snapshot.snapshot_id,
            model_used=None,
            limited_mode=snapshot.retriever.limited_mode,
        )

    def _first_enabled_provider(self):
        if self.primary and self.primary.enabled:
            return self.primary
        if self.fallback and self.fallback.enabled:
            return self.fallback
        return None

    def _fallback_if_allowed(self, provider, error: ProviderError):
        if not error.retryable:
            return None
        if provider is self.primary and self.fallback and self.fallback.enabled:
            return self.fallback
        return None

    def _get_user_lock(self, user_id: str):
        with self._lock:
            return self._user_locks.setdefault(user_id, threading.Lock())

    def _cache_key(self, *, conversation_id, question, selected, snapshot_id):
        prompt_version = "v1"
        model = (
            self.primary.model_id
            if self.primary and self.primary.enabled
            else self.fallback.model_id
            if self.fallback and self.fallback.enabled
            else "search-only"
        )
        raw = json.dumps(
            {
                "conversation_id": conversation_id,
                "question": question,
                "selected": sorted(selected),
                "snapshot": snapshot_id,
                "prompt": prompt_version,
                "model": model,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _save_response(self, identity, request, response: ChatResponse):
        assistant_message = Message(
            message_id=response.message_id,
            role="assistant",
            text=response.answer_th,
            created_at=utc_now_iso(),
            citations=response.citations,
            status=response.status.value,
            client_message_id=request.client_message_id,
            selected_source_ids=request.selected_source_ids,
            clarification_questions=response.clarification_questions,
        )
        self.store.add_message(request.conversation_id, identity, assistant_message)
        return response

    def _response_from_message(self, message: Message, cache_hit: bool):
        return ChatResponse(
            message_id=message.message_id,
            status=AnswerStatus(message.status or AnswerStatus.ANSWER.value),
            answer_th=message.text,
            citations=message.citations,
            clarification_questions=message.clarification_questions,
            cache_hit=cache_hit,
        )