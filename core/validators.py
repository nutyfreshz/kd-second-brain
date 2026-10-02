from __future__ import annotations

from dataclasses import dataclass

from core.schemas import Claim, Citation, EvidenceChunk


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]


def validate_answer(
    *,
    claims: list[Claim],
    citations: list[Citation],
    evidence: list[EvidenceChunk],
    user_text: str,
) -> ValidationResult:
    """Validate grounding links without rejecting valid paraphrases/derived summaries.

    The model may paraphrase, count items, or restate numbers with different formatting.
    Safety is enforced by requiring every factual claim to point only to citations that
    originate from the retrieved evidence set. Citation quote text itself is populated
    from trusted evidence by the parser, not trusted from model output.
    """
    _ = user_text
    errors: list[str] = []
    evidence_ids = {c.chunk_id for c in evidence}
    citation_ids = {c.chunk_id for c in citations}

    for citation in citations:
        if citation.chunk_id not in evidence_ids:
            errors.append(f"unknown_citation:{citation.chunk_id}")

    for claim in claims:
        if not claim.citation_ids:
            errors.append("claim_without_citation")
            continue
        unknown = [cid for cid in claim.citation_ids if cid not in citation_ids]
        if unknown:
            errors.append("claim_unknown_citation:" + ",".join(unknown))

    return ValidationResult(valid=not errors, errors=tuple(errors))
