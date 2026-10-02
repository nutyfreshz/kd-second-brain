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
    rendered = []
    for chunk in evidence:
        rendered.append(
            f"<evidence chunk_id=\"{chunk.chunk_id}\" source_id=\"{chunk.source_id}\">\n"
            f"TITLE: {chunk.title}\nHEADING: {chunk.heading}\n{chunk.text}\n</evidence>"
        )
    return (
        "CONVERSATION CONTEXT (user messages only; use for reference resolution, "
        "not as policy evidence):\n"
        f"{conversation_context or '(none)'}\n\n"
        "CURRENT QUESTION:\n"
        f"{question}\n\n"
        "RETRIEVED EVIDENCE:\n"
        + "\n\n".join(rendered)
    )


def parse_model_json(raw: str, evidence: list[EvidenceChunk]) -> dict[str, Any]:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("model_output_not_json")
        data = json.loads(m.group(0))

    status_raw = str(data.get("status") or "").strip().lower()
    if status_raw not in ALLOWED_MODEL_STATUSES:
        raise ValueError("invalid_model_status")
    status = ALLOWED_MODEL_STATUSES[status_raw]

    by_chunk = {c.chunk_id: c for c in evidence}
    citations: list[Citation] = []
    for item in data.get("citations") or []:
        cid = str(item.get("chunk_id") or "")
        quote = str(item.get("quote") or "")
        source = by_chunk.get(cid)
        if source is None:
            # Keep invalid ID; validator will fail closed.
            citations.append(
                Citation(
                    chunk_id=cid,
                    quote=quote,
                    source_id="",
                    title="",
                    heading="",
                )
            )
        else:
            citations.append(
                Citation(
                    chunk_id=cid,
                    quote=quote,
                    source_id=source.source_id,
                    title=source.title,
                    heading=source.heading,
                )
            )

    claims = [
        Claim(
            text=str(item.get("text") or ""),
            citation_ids=tuple(str(x) for x in (item.get("citation_ids") or [])),
        )
        for item in (data.get("claims") or [])
        if str(item.get("text") or "").strip()
    ]
    clarifications = [
        str(x).strip()
        for x in (data.get("clarification_questions") or [])
        if str(x).strip()
    ][:2]
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
