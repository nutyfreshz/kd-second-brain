from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Protocol

from core.md_parser import ParsedDocument, is_active_status, is_effective, parse_markdown
from core.text_extraction import extract_text_from_path, is_markdown_name
from core.retrieval import HybridRetriever
from core.schemas import EvidenceChunk, SourceMeta


@dataclass(frozen=True)
class RawKnowledgeFile:
    source_id: str
    text: str
    updated_at: str
    origin_url: str | None = None
    title_hint: str | None = None
    use_frontmatter: bool = True


class KnowledgeSource(Protocol):
    def read_all(self) -> list[RawKnowledgeFile]: ...


class LocalKnowledgeSource:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def read_all(self) -> list[RawKnowledgeFile]:
        if not self.directory.exists():
            return []
        files: list[RawKnowledgeFile] = []
        for path in sorted(p for p in self.directory.rglob("*") if p.is_file()):
            try:
                text = extract_text_from_path(path)
            except Exception:
                continue
            if not text or not text.strip():
                continue
            stat = path.stat()
            rel = str(path.relative_to(self.directory)).replace("\\", "/")
            files.append(
                RawKnowledgeFile(
                    source_id=rel,
                    text=text,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    origin_url=None,
                    title_hint=path.name,
                    use_frontmatter=is_markdown_name(path.name),
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
        self._check_lock = threading.Lock()
        self._snapshot: KnowledgeSnapshot | None = None
        self._source_fingerprint: str | None = None
        self._last_manifest_check = 0.0

    @property
    def snapshot(self) -> KnowledgeSnapshot | None:
        return self._snapshot

    def sync(self) -> KnowledgeSnapshot:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("knowledge_sync_already_running")
        try:
            raw_files = self.source.read_all()
            self._source_fingerprint = getattr(
                self.source, "last_manifest_fingerprint", self._source_fingerprint
            )
            parsed: list[ParsedDocument] = []
            for raw in raw_files:
                doc = parse_markdown(
                    raw.source_id,
                    raw.text,
                    updated_at=raw.updated_at,
                    origin_url=raw.origin_url,
                    title_hint=raw.title_hint,
                    use_frontmatter=raw.use_frontmatter,
                )
                if is_active_status(doc.meta.status):
                    parsed.append(doc)
            sources = tuple(p.meta for p in parsed)
            chunks = tuple(c for p in parsed for c in p.chunks)
            conflicts = tuple(_detect_authority_conflicts(sources))
            digest = hashlib.sha256(
                json.dumps(sorted((s.source_id, s.content_hash, s.origin_url or "", s.updated_at) for s in sources)).encode("utf-8")
            ).hexdigest()[:16]
            if self._snapshot and self._snapshot.snapshot_id == digest:
                return self._snapshot
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

    def maybe_sync(self, min_interval_seconds: int = 180) -> KnowledgeSnapshot:
        if self._snapshot is None:
            return self.sync()
        if min_interval_seconds <= 0:
            return self._snapshot

        now = time.monotonic()
        with self._check_lock:
            if now - self._last_manifest_check < min_interval_seconds:
                return self._snapshot
            self._last_manifest_check = now

            fingerprint_fn = getattr(self.source, "manifest_fingerprint", None)
            if not callable(fingerprint_fn):
                return self._snapshot
            try:
                fingerprint = fingerprint_fn()
            except Exception:
                # Keep the last good snapshot when Drive metadata cannot be checked.
                return self._snapshot

            if fingerprint == self._source_fingerprint:
                return self._snapshot

            try:
                return self.sync()
            except RuntimeError:
                # Another sync may already be building the next atomic snapshot.
                return self._snapshot


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