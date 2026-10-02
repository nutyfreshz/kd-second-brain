from __future__ import annotations

import os
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from adapters.conversation_store import InMemoryConversationStore
from adapters.drive import GoogleDriveKnowledgeSource
from adapters.identity import identity_from_headers
from adapters.providers.gemini import GeminiProvider
from adapters.providers.openrouter import OpenRouterProvider
from core.chat_service import ChatService
from core.kb_sync import KnowledgeManager, LocalKnowledgeSource
from core.schemas import ChatRequest
from settings import Settings


st.set_page_config(page_title="AIS Knowledge Notebook", page_icon="📚", layout="wide")


@st.cache_resource
def build_service() -> tuple[Settings, ChatService]:
    settings = Settings.from_env()
    if settings.knowledge_source == "drive":
        source = GoogleDriveKnowledgeSource(
            settings.drive_published_folder_id or "",
            service_account_json=settings.google_service_account_json,
            service_account_file=settings.google_service_account_file,
        )
    else:
        source = LocalKnowledgeSource(settings.local_knowledge_dir)

    knowledge = KnowledgeManager(
        source,
        semantic_enabled=settings.enable_semantic,
        embedding_model=settings.embedding_model,
    )
    try:
        knowledge.sync()
    except Exception:
        pass

    service = ChatService(
        knowledge=knowledge,
        store=InMemoryConversationStore(),
        primary=GeminiProvider(
            settings.gemini_api_key,
            settings.gemini_model,
            free_verified=settings.gemini_free_verified,
        ),
        fallback=OpenRouterProvider(
            settings.openrouter_api_key,
            settings.openrouter_model,
            free_verified=settings.openrouter_free_verified,
            paid_allowed=settings.openrouter_paid_allowed,
        ),
        max_concurrent_inference=settings.max_concurrent_inference,
        system_prompt_path=str(ROOT / "prompts" / "SYSTEM_PROMPT_TH.md"),
        allow_external_llm_for_all_evidence=(
            settings.knowledge_source == "drive"
            and settings.drive_external_llm_allow_all
        ),
    )
    return settings, service


settings, service = build_service()

try:
    headers = st.context.headers
except Exception:
    headers = {}

try:
    identity = identity_from_headers(headers, settings)
except PermissionError:
    st.error("ไม่พบ Databricks user identity และ anonymous mode ถูกปิด")
    st.stop()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = service.create_conversation(identity)

snapshot = service.knowledge.snapshot
sources = service.list_sources(identity)

with st.sidebar:
    st.title("AIS Knowledge")
    st.caption(f"ผู้ใช้: {identity.username}")
    st.caption(f"ค้นอัตโนมัติจากคลังความรู้ทั้งหมด · {len(sources)} sources")

    if st.button("New chat", use_container_width=True):
        st.session_state.conversation_id = service.create_conversation(identity)
        st.rerun()

    if identity.is_admin:
        if st.button("Sync knowledge", use_container_width=True):
            try:
                snapshot = service.sync_knowledge(identity)
                st.success(f"Sync สำเร็จ: {len(snapshot.sources)} files")
                st.rerun()
            except Exception:
                st.error("Sync ไม่สำเร็จ โปรดตรวจ source/credential/connectivity")

        with st.expander("Admin health"):
            st.write(f"Knowledge source: `{settings.knowledge_source}`")
            st.write(
                "Gemini: "
                + ("enabled" if service.primary and service.primary.enabled else "disabled / not verified")
            )
            st.write(
                "OpenRouter fallback: "
                + ("enabled" if service.fallback and service.fallback.enabled else "disabled / not verified")
            )
            st.write(f"OpenRouter model: `{settings.openrouter_model}`")
            st.write(
                "Paid model access: "
                + ("allowed" if settings.openrouter_paid_allowed else "disabled")
            )
            st.write("Free quota: ยังไม่ยืนยันจนกว่าจะ run credential check")
            st.write(
                "Drive external LLM policy: "
                + ("allow all synced knowledge" if settings.drive_external_llm_allow_all else "per-document")
            )
            if snapshot:
                st.write(f"Published sources: {len(snapshot.sources)}")
                st.write(f"Authority conflicts: {len(snapshot.conflicts)}")

    if snapshot:
        st.caption(f"Snapshot: {snapshot.snapshot_id}")
        st.caption(f"Last sync: {snapshot.created_at}")
        if snapshot.retriever.limited_mode:
            st.warning("Semantic retrieval ไม่พร้อม ขณะนี้ใช้ lexical search เป็นหลัก")
    else:
        st.warning("ยังไม่มี active knowledge snapshot")

st.title("AIS Knowledge Notebook")
st.caption("ถาม ตอบ เปรียบเทียบ และเชื่อมโยงจากคลังความรู้ที่ผ่านการ review")

if not sources:
    st.info(
        "ยังไม่มีเอกสาร published ที่พร้อมใช้ กรุณาเพิ่ม reviewed MD ใน source "
        "หรือให้ Admin ตรวจ Google Drive configuration"
    )
    st.stop()

conversation = service.get_conversation(identity, st.session_state.conversation_id)
for msg in conversation.messages:
    with st.chat_message("user" if msg.role == "user" else "assistant"):
        st.markdown(msg.text)
        if msg.role == "assistant" and msg.clarification_questions:
            for question in msg.clarification_questions:
                st.markdown(f"• {question}")
        if msg.role == "assistant" and msg.citations:
            with st.expander(f"หลักฐาน {len(msg.citations)} รายการ"):
                for citation in msg.citations:
                    st.markdown(f"**{citation.title} · {citation.heading}**")
                    st.code(citation.quote, language=None)

prompt = st.chat_input("ถามจากคลังความรู้...")
if prompt:
    client_message_id = str(uuid.uuid4())
    response = service.send_message(
        identity,
        ChatRequest(
            conversation_id=st.session_state.conversation_id,
            client_message_id=client_message_id,
            text=prompt,
            selected_source_ids=(),
        ),
    )
    st.rerun()