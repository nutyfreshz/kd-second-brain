from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import threading
from typing import Protocol

from core.md_parser import ParsedDocument, is_effective, parse_markdown
from core.retrieval import HybridRetriever
from core.schemas import EvidenceChunk, SourceMeta


@dataclass(frozen=True)
class RawKnowledgeFile:
    source_id: str
    text: str
    updated_at: str
    origin_url: str | None = None


class KnowledgeSource(Protocol):
    def read_all(self) -> list[RawKnowledgeFile]: ...


class LocalKnowledgeSource:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def read_all(self) -> list[RawKnowledgeFile]:
        if not self.directory.exists():
            return []
        files: list[RawKnowledgeFile] = []
        for path in sorted(self.directory.glob("*.md")):
            stat = path.stat()
            files.append(
                RawKnowledgeFile(
                    source_id=path.name,
                    text=path.read_text(encoding="utf-8"),
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    origin_url=None,
                )
            )
        return files


@dataclass(frozen=True)
class KnowledgeSnapshot:
    snapshot_id: str
    created_at: str
    sources: tuple[SourceMeta, ...]
    chunks: tuple[EvidenceChunk, ...]
    retriever: HybridRetriever
    conflicts: tuple[str, ...] = ()


class KnowledgeManager:
    def __init__(self, source: KnowledgeSource, *, semantic_enabled: bool, embedding_model: str):
        self.source = source
        self.semantic_enabled = semantic_enabled
        self.embedding_model = embedding_model
        self._lock = threading.Lock()
        self._snapshot: KnowledgeSnapshot | None = None

    @property
    def snapshot(self) -> KnowledgeSnapshot | None:
        return self._snapshot

    def sync(self) -> KnowledgeSnapshot:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("knowledge_sync_already_running")
        try:
            raw_files = self.source.read_all()
            parsed: list[ParsedDocument] = []
            for raw in raw_files:
                doc = parse_markdown(raw.source_id, raw.text, updated_at=raw.updated_at, origin_url=raw.origin_url)
                if doc.meta.status == "published":
                    parsed.append(doc)
            sources = tuple(p.meta for p in parsed)
            chunks = tuple(c for p in parsed for c in p.chunks)
            conflicts = tuple(_detect_authority_conflicts(sources))
            digest = hashlib.sha256(
                "".join(sorted(s.content_hash for s in sources)).encode("utf-8")
            ).hexdigest()[:16]
            snapshot = KnowledgeSnapshot(
                snapshot_id=digest or "empty",
                created_at=datetime.now(timezone.utc).isoformat(),
                sources=sources,
                chunks=chunks,
                retriever=HybridRetriever(
                    chunks,
                    semantic_enabled=self.semantic_enabled,
                    model_name=self.embedding_model,
                ),
                conflicts=conflicts,
            )
            self._snapshot = snapshot
            return snapshot
        finally:
            self._lock.release()


def _detect_authority_conflicts(sources: tuple[SourceMeta, ...]) -> list[str]:
    by_doc: dict[str, list[SourceMeta]] = {}
    for source in sources:
        by_doc.setdefault(source.doc_id, []).append(source)
    conflicts: list[str] = []
    for doc_id, group in by_doc.items():
        if len(group) <= 1:
            continue
        ids_and_versions = {item.source_id for item in group} | {item.version for item in group}
        superseded = set()
        for item in group:
            superseded.update(x for x in item.supersedes if x in ids_and_versions)
        authoritative = [
            item for item in group
            if item.source_id not in superseded and item.version not in superseded
        ]
        if len(authoritative) != 1:
            conflicts.append(doc_id)
    return conflicts


def resolve_authoritative_sources(
    sources: tuple[SourceMeta, ...], as_of: date
) -> tuple[set[str], tuple[str, ...]]:
    effective = [s for s in sources if is_effective(s, as_of)]
    by_doc: dict[str, list[SourceMeta]] = {}
    for source in effective:
        by_doc.setdefault(source.doc_id, []).append(source)

    allowed: set[str] = set()
    conflicts: list[str] = []
    for doc_id, group in by_doc.items():
        if len(group) == 1:
            allowed.add(group[0].source_id)
            continue
        identifiers = {s.source_id for s in group} | {s.version for s in group}
        superseded: set[str] = set()
        for source in group:
            superseded.update(x for x in source.supersedes if x in identifiers)
        candidates = [
            s for s in group
            if s.source_id not in superseded and s.version not in superseded
        ]
        if len(candidates) == 1:
            allowed.add(candidates[0].source_id)
        else:
            conflicts.append(doc_id)
    return allowed, tuple(conflicts)
