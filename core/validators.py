from __future__ import annotations

from dataclasses import dataclass
import re

from core.schemas import Claim, Citation, EvidenceChunk

_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d{1,4}(?:[-/]\d{1,2}){1,2}|\d+(?:,\d{3})*(?:\.\d+)?%?)(?![\w])"
)

def normalize_ws(text: str) -> str:
    return " ".join(text.split())

@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]

def validate_answer(*, claims: list[Claim], citations: list[Citation], evidence: list[EvidenceChunk], user_text: str) -> ValidationResult:
    errors: list[str] = []
    by_chunk = {c.chunk_id: c for c in evidence}
    citation_ids = {c.chunk_id for c in citations}
    for citation in citations:
        source = by_chunk.get(citation.chunk_id)
        if source is None:
            errors.append(f"unknown_citation:{citation.chunk_id}")
            continue
        quote = normalize_ws(citation.quote)
        source_text = normalize_ws(source.text)
        if not quote or quote not in source_text:
            errors.append(f"quote_not_in_source:{citation.chunk_id}")
    for claim in claims:
        if not claim.citation_ids:
            errors.append("claim_without_citation")
            continue
        unknown = [cid for cid in claim.citation_ids if cid not in citation_ids]
        if unknown:
            errors.append("claim_unknown_citation:" + ",".join(unknown))
        evidence_text = " ".join(by_chunk[cid].text for cid in claim.citation_ids if cid in by_chunk)
        permitted_numbers = set(_NUMBER_RE.findall(evidence_text))
        permitted_numbers.update(_NUMBER_RE.findall(user_text))
        for number in _NUMBER_RE.findall(claim.text):
            if number not in permitted_numbers:
                errors.append(f"unsupported_number:{number}")
    return ValidationResult(valid=not errors, errors=tuple(errors))
