from __future__ import annotations

from dataclasses import dataclass
import re
from decimal import Decimal

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
    """Check citation integrity and critical numeric literals, not semantic entailment.

    A valid reference alone cannot prove that a paraphrase follows from its source.
    Derived counts without units are allowed; unsupported percentages, currency and
    time limits fail closed. Live answer-quality evaluation remains necessary.
    """
    _ = user_text
    errors: list[str] = []
    by_id = {c.chunk_id: c for c in evidence}
    evidence_ids = set(by_id)
    if not claims:
        errors.append("answer_without_claims")
    citation_ids = {c.chunk_id for c in citations}

    for citation in citations:
        if citation.chunk_id not in evidence_ids:
            errors.append(f"unknown_citation:{citation.chunk_id}")
        else:
            source = by_id[citation.chunk_id]
            if (citation.source_id != source.source_id or not citation.quote.strip()
                    or citation.quote not in source.text):
                errors.append(f"citation_not_verbatim:{citation.chunk_id}")

    for claim in claims:
        if not claim.text.strip():
            errors.append("empty_claim")
        if not claim.citation_ids:
            errors.append("claim_without_citation")
            continue
        unknown = [cid for cid in claim.citation_ids if cid not in citation_ids]
        if unknown:
            errors.append("claim_unknown_citation:" + ",".join(unknown))

        support = "\n".join(by_id[cid].text for cid in claim.citation_ids if cid in by_id)
        unsupported = _critical_numbers(claim.text) - _critical_numbers(support)
        if unsupported:
            errors.append("unsupported_critical_number")

    return ValidationResult(valid=not errors, errors=tuple(errors))


_UNIT_ALIASES = {
    "%": "%", "percent": "%", "เปอร์เซ็นต์": "%",
    "บาท": "thb", "thb": "thb", "฿": "thb",
    "usd": "usd", "$": "usd", "ดอลลาร์": "usd",
    "day": "day", "days": "day", "วัน": "day",
    "hour": "hour", "hours": "hour", "ชั่วโมง": "hour",
    "point": "point", "points": "point", "คะแนน": "point",
}
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CRITICAL = re.compile(r"(" + _NUMBER + r")\s*(%|percent\b|เปอร์เซ็นต์|บาท|thb\b|usd\b|ดอลลาร์|days?\b|วัน|hours?\b|ชั่วโมง|points?\b|คะแนน)", re.I)
_CURRENCY = re.compile(r"([฿$])\s*(" + _NUMBER + r")")


def _critical_numbers(text: str) -> set[tuple[Decimal, str]]:
    text = text.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    values = {(Decimal(number.replace(",", "")), _UNIT_ALIASES[unit.lower()])
              for number, unit in _CRITICAL.findall(text)}
    values.update((Decimal(number.replace(",", "")), _UNIT_ALIASES[unit])
                  for unit, number in _CURRENCY.findall(text))
    return values
