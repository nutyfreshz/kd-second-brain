from __future__ import annotations

import json
import re
from typing import Any

from core.schemas import AnswerStatus, Claim, Citation, EvidenceChunk


ALLOWED_MODEL_STATUSES = {
    "answer": AnswerStatus.ANSWER,
    "clarify": AnswerStatus.CLARIFY,
    "not_found": AnswerStatus.NOT_FOUND,
    "conflict": AnswerStatus.CONFLICT,
}


def build_evidence_prompt(
    question: str,
    evidence: list[EvidenceChunk],
    conversation_context: str,
) -> str:
    # JSON keeps source text separate from delimiters. Content remains untrusted.
    return json.dumps({
        "conversation_context_for_reference_only": conversation_context,
        "current_question": question,
        "retrieved_evidence_untrusted": [
            {"chunk_id": c.chunk_id, "source_id": c.source_id,
             "title": c.title, "heading": c.heading, "text": c.text}
            for c in evidence
        ],
    }, ensure_ascii=False)


def parse_model_json(raw: str, evidence: list[EvidenceChunk]) -> dict[str, Any]:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("model_output_not_json")
        data = json.loads(m.group(0))

    if not isinstance(data, dict):
        raise ValueError("model_output_not_object")
    for field in ("claims", "clarification_questions", "citations"):
        if field in data and not isinstance(data[field], list):
            raise ValueError("invalid_" + field)
    for item in data.get("claims", []):
        if (not isinstance(item, dict) or not isinstance(item.get("text"), str)
                or not isinstance(item.get("citation_ids"), list)
                or not all(isinstance(cid, str) for cid in item["citation_ids"])):
            raise ValueError("invalid_claim")
    for item in data.get("citations", []):
        if not isinstance(item, dict):
            raise ValueError("invalid_citation")
    if not all(isinstance(q, str) for q in data.get("clarification_questions", [])):
        raise ValueError("invalid_clarification_questions")
    if not isinstance(data.get("answer_th"), str):
        raise ValueError("invalid_answer_text")
    status_raw = str(data.get("status") or "").strip().lower()
    if status_raw not in ALLOWED_MODEL_STATUSES:
        raise ValueError("invalid_model_status")
    status = ALLOWED_MODEL_STATUSES[status_raw]

    claims = [
        Claim(
            text=str(item.get("text") or ""),
            citation_ids=tuple(str(x) for x in (item.get("citation_ids") or [])),
        )
        for item in (data.get("claims") or [])
        if str(item.get("text") or "").strip()
    ]

    # Citation IDs are authored once, on claims. The backend derives the display
    # citations directly from retrieved evidence, avoiding model-side drift between
    # claims[] and a second citations[] structure.
    requested_ids: list[str] = []
    for claim in claims:
        for cid in claim.citation_ids:
            if cid not in requested_ids:
                requested_ids.append(cid)

    # Tolerate the old top-level citations shape during rolling deployments.
    for item in data.get("citations") or []:
        cid = str(item.get("chunk_id") or "")
        if cid and cid not in requested_ids:
            requested_ids.append(cid)

    by_chunk = {c.chunk_id: c for c in evidence}
    citations: list[Citation] = []
    for cid in requested_ids:
        source = by_chunk.get(cid)
        if source is None:
            citations.append(
                Citation(
                    chunk_id=cid,
                    quote="",
                    source_id="",
                    title="",
                    heading="",
                )
            )
            continue
        trusted_quote = source.text.strip()
        citations.append(
            Citation(
                chunk_id=cid,
                quote=trusted_quote,
                source_id=source.source_id,
                title=source.title,
                heading=source.heading,
            )
        )

    clarifications = [
        str(x).strip()
        for x in (data.get("clarification_questions") or [])
        if str(x).strip()
    ][:2]
    if status == AnswerStatus.ANSWER and not claims:
        raise ValueError("answer_without_claims")
    if status == AnswerStatus.CLARIFY and not clarifications:
        raise ValueError("clarify_without_question")
    return {
        "status": status,
        "answer_th": str(data.get("answer_th") or "").strip(),
        "claims": claims,
        "citations": citations,
        "clarification_questions": clarifications,
    }


def evidence_only_answer(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return "ไม่พบหลักฐานที่เกี่ยวข้องในคลังที่เลือก"
    return (
        "AI synthesis ยังไม่พร้อม แต่พบหลักฐานที่เกี่ยวข้องด้านล่าง "
        "กรุณาเปิด citation เพื่อตรวจข้อความต้นฉบับ"
    )


def render_grounded_claims(claims: list[Claim], citations: list[Citation]) -> str:
    """Display exactly the checked claims, avoiding a second unvalidated answer."""
    numbers = {citation.chunk_id: i for i, citation in enumerate(citations, 1)}
    return "\n\n".join(
        claim.text + " " + " ".join(f"[{numbers[cid]}]" for cid in dict.fromkeys(claim.citation_ids))
        for claim in claims
    )
