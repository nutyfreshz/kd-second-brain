from __future__ import annotations

from datetime import date
import re

from core.schemas import Conversation


_THAI_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1,
    "กุมภาพันธ์": 2, "ก.พ.": 2,
    "มีนาคม": 3, "มี.ค.": 3,
    "เมษายน": 4, "เม.ย.": 4,
    "พฤษภาคม": 5, "พ.ค.": 5,
    "มิถุนายน": 6, "มิ.ย.": 6,
    "กรกฎาคม": 7, "ก.ค.": 7,
    "สิงหาคม": 8, "ส.ค.": 8,
    "กันยายน": 9, "ก.ย.": 9,
    "ตุลาคม": 10, "ต.ค.": 10,
    "พฤศจิกายน": 11, "พ.ย.": 11,
    "ธันวาคม": 12, "ธ.ค.": 12,
}


def resolve_effective_date(text: str, previous: str | None = None) -> date:
    m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])(?:[-/](0?[1-9]|[12]\d|3[01]))?\b", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    m = re.search(r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(20\d{2}|25\d{2})\b", text)
    if m:
        year = int(m.group(3))
        if year >= 2400:
            year -= 543
        return date(year, int(m.group(2)), int(m.group(1)))
    for label, month in _THAI_MONTHS.items():
        m = re.search(re.escape(label) + r"\s*(20\d{2}|25\d{2})", text)
        if m:
            year = int(m.group(1))
            if year >= 2400:
                year -= 543
            return date(year, month, 1)
    if previous:
        try:
            return date.fromisoformat(previous)
        except ValueError:
            pass
    return date.today()


def retrieval_query(conversation: Conversation, current_text: str) -> tuple[str, str]:
    user_messages = [m.text for m in conversation.messages if m.role == "user"]
    previous = user_messages[-3:]
    context = "\n".join(previous)
    compact = current_text.strip()
    if len(compact) <= 90 and previous:
        query = "\n".join(previous[-2:] + [compact])
    else:
        query = compact
    return query, context
