from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import re
from typing import Iterable

from core.schemas import EvidenceChunk, SourceMeta

_FRONTMATTER_BOUNDARY = "---"

@dataclass(frozen=True)
class ParsedDocument:
    meta: SourceMeta
    content: str
    chunks: tuple[EvidenceChunk, ...]

def _simple_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
    return value.strip("'\"")

def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith(_FRONTMATTER_BOUNDARY + "\n"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_BOUNDARY:
            end = i
            break
    if end is None:
        return {}, text
    meta: dict[str, object] = {}
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        meta[key.strip()] = _simple_scalar(value)
    return meta, "\n".join(lines[end + 1 :]).strip()

def _split_sections(content: str) -> list[tuple[str, str]]:
    current_heading = "Document"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    def flush():
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))
        buffer = []
    for line in content.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            current_heading = m.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return sections or [("Document", content.strip())]

def _chunk_section(heading: str, body: str, max_chars: int = 3200) -> Iterable[str]:
    blocks = re.split(r"\n\s*\n", body)
    current: list[str] = []
    size = 0
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if current and size + len(b) + 2 > max_chars:
            yield "\n\n".join(current)
            current, size = [], 0
        if len(b) > max_chars and not current:
            lines = b.splitlines()
            piece: list[str] = []
            psize = 0
            for line in lines:
                if piece and psize + len(line) + 1 > max_chars:
                    yield "\n".join(piece)
                    piece, psize = [], 0
                piece.append(line)
                psize += len(line) + 1
            if piece:
                yield "\n".join(piece)
            continue
        current.append(b)
        size += len(b) + 2
    if current:
        yield "\n\n".join(current)

def parse_markdown(source_id: str, text: str, *, updated_at: str = "", origin_url: str | None = None) -> ParsedDocument:
    raw_meta, content = parse_frontmatter(text)
    title = str(raw_meta.get("title") or "").strip()
    if not title:
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    title = title or source_id
    doc_id = str(raw_meta.get("doc_id") or source_id)
    version = str(raw_meta.get("version") or "unversioned")
    status = str(raw_meta.get("status") or "published").lower()
    supersedes_raw = raw_meta.get("supersedes") or []
    aliases_raw = raw_meta.get("aliases") or []
    if isinstance(supersedes_raw, str):
        supersedes_raw = [supersedes_raw]
    if isinstance(aliases_raw, str):
        aliases_raw = [aliases_raw]
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta = SourceMeta(
        source_id=source_id, doc_id=doc_id, title=title, version=version, status=status,
        updated_at=updated_at,
        effective_from=str(raw_meta.get("effective_from") or "") or None,
        effective_to=str(raw_meta.get("effective_to") or "") or None,
        supersedes=tuple(str(x) for x in supersedes_raw),
        aliases=tuple(str(x) for x in aliases_raw),
        external_llm_allowed=bool(raw_meta.get("external_llm_allowed", False)),
        scope=str(raw_meta.get("scope") or "") or None,
        origin_url=origin_url,
        content_hash=content_hash,
    )
    chunks: list[EvidenceChunk] = []
    ordinal = 0
    for heading, body in _split_sections(content):
        for piece in _chunk_section(heading, body):
            chunk_id = f"{source_id}:{ordinal}"
            chunks.append(EvidenceChunk(
                chunk_id=chunk_id, source_id=source_id, title=title,
                heading=heading, text=piece, ordinal=ordinal
            ))
            ordinal += 1
    return ParsedDocument(meta=meta, content=content, chunks=tuple(chunks))

def is_effective(meta: SourceMeta, as_of: date | None) -> bool:
    if meta.status != "published":
        return False
    if as_of is None:
        return True
    try:
        if meta.effective_from and as_of < date.fromisoformat(meta.effective_from):
            return False
        if meta.effective_to and as_of > date.fromisoformat(meta.effective_to):
            return False
    except ValueError:
        return False
    return True
