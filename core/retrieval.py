from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable
from functools import lru_cache
import threading

from core.schemas import EvidenceChunk


_TOKEN_RE = re.compile(r"(?:[A-Za-z][A-Za-z0-9]*|\d+[A-Za-z][A-Za-z0-9]*)(?:[-_/][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?|[\u0E00-\u0E7F]+", re.UNICODE)

_OVERVIEW_RE = re.compile(
    r"(สรุป|ภาพรวม|โดยรวม|ทั้งหมด|ทั้งคลัง|เปรียบเทียบ|ต่างกัน|"
    r"คิดยังไง|ทำงานยังไง|อธิบาย.*รวม|overview|summary|summarize|"
    r"compare|comparison|how .* work)",
    re.I,
)
_GLOBAL_OVERVIEW_RE = re.compile(
    r"(ทั้งหมด|ทั้งคลัง|ทุกเอกสาร|ทุก source|all sources|whole corpus|entire corpus)",
    re.I,
)


def is_overview_query(query: str) -> bool:
    return bool(_OVERVIEW_RE.search(query or ""))


def is_global_overview_query(query: str) -> bool:
    return bool(_GLOBAL_OVERVIEW_RE.search(query or ""))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u0E00-\u0E7F]+", token):
            compact = token.strip()
            if len(compact) <= 3:
                tokens.append(compact)
            else:
                tokens.extend(compact[i : i + 3] for i in range(len(compact) - 2))
        else:
            tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, chunks: Iterable[EvidenceChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(c.text + " " + c.heading + " " + c.title) for c in self.chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(self.docs):
            counts = Counter(doc)
            self.tf.append(counts)
            self.df.update(counts.keys())
            for term in counts:
                self.postings[term].append(i)

    def search(self, query: str, top_k: int = 12, *,
               allowed_source_ids: set[str] | None = None) -> list[tuple[EvidenceChunk, float]]:
        if not self.chunks or top_k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        n = len(self.chunks)
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in self.postings.get(term, ()):
                if allowed_source_ids is not None and self.chunks[i].source_id not in allowed_source_ids:
                    continue
                freq = self.tf[i][term]
                dl = self.lengths[i] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (freq * (self.k1 + 1) / denom)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [(self.chunks[i], scores[i]) for i in ordered]


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class SemanticIndex:
    def __init__(self, chunks: Iterable[EvidenceChunk], model_name: str):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.available = False
        self.error: str | None = None
        self._model = None
        self._vectors = None
        if not self.chunks:
            return
        try:
            self._model = _load_embedding_model(model_name)
            passages = [f"{c.title}\n{c.heading}\n{c.text}" for c in self.chunks]
            # Encode every token window: a long Thai paragraph must not lose its tail.
            tokenizer = self._model.tokenizer
            limit = min(int(self._model.max_seq_length), 512) - 16
            windows, self._owners = [], []
            for i, passage in enumerate(passages):
                ids = tokenizer.encode(passage, add_special_tokens=False)
                for start in range(0, max(1, len(ids)), max(1, limit - 48)):
                    windows.append("passage: " + tokenizer.decode(ids[start:start + limit], skip_special_tokens=True))
                    self._owners.append(i)
                    if start + limit >= len(ids):
                        break
            self._vectors = self._model.encode(
                windows, normalize_embeddings=True, show_progress_bar=False
            )
            self._query_lock = threading.Lock()
            self.available = True
        except Exception as exc:
            self.error = type(exc).__name__

    def search(self, query: str, top_k: int = 12, *,
               allowed_source_ids: set[str] | None = None) -> list[tuple[EvidenceChunk, float]]:
        if not self.available or self._model is None or self._vectors is None or top_k <= 0:
            return []
        import numpy as np
        with self._query_lock:
            q = self._model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        scores = np.dot(self._vectors, q)
        best: dict[int, float] = {}
        for row, score in enumerate(scores):
            i = self._owners[row]
            if allowed_source_ids is not None and self.chunks[i].source_id not in allowed_source_ids:
                continue
            best[i] = max(best.get(i, -1.0), float(score))
        indices = sorted(best, key=lambda i: (-best[i], i))[:top_k]
        return [(self.chunks[i], best[i]) for i in indices]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[EvidenceChunk, float]]],
    top_k: int = 12,
    rrf_k: int = 60,
) -> list[EvidenceChunk]:
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, EvidenceChunk] = {}
    for ranking in rankings:
        for rank, (chunk, _raw_score) in enumerate(ranking, 1):
            fused[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            by_id[chunk.chunk_id] = chunk
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        EvidenceChunk(**{**by_id[cid].__dict__, "score": score})
        for cid, score in ordered
    ]


class HybridRetriever:
    def __init__(self, chunks: Iterable[EvidenceChunk], *, semantic_enabled: bool, model_name: str):
        self.chunks = list(chunks)
        self.lexical = BM25Index(self.chunks)
        self.semantic = SemanticIndex(self.chunks, model_name) if semantic_enabled else None
        self.by_source: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in self.chunks:
            self.by_source[chunk.source_id].append(chunk)
        for group in self.by_source.values():
            group.sort(key=lambda chunk: chunk.ordinal)

    @property
    def limited_mode(self) -> bool:
        return self.semantic is None or not self.semantic.available

    def _allowed(self, source_id: str, allowed_source_ids: set[str] | None) -> bool:
        return allowed_source_ids is None or source_id in allowed_source_ids

    def _representative_chunks(
        self,
        allowed_source_ids: set[str] | None,
        limit: int,
    ) -> list[EvidenceChunk]:
        source_ids = sorted(
            sid for sid in self.by_source
            if self._allowed(sid, allowed_source_ids)
        )
        out: list[EvidenceChunk] = []
        for sid in source_ids:
            group = self.by_source.get(sid) or []
            if group:
                out.append(group[0])
            if len(out) >= limit:
                break
        return out

    def _diversify(
        self,
        ranked: list[EvidenceChunk],
        *,
        limit: int,
        max_per_source: int,
    ) -> list[EvidenceChunk]:
        if limit <= 0:
            return []
        groups: dict[str, list[EvidenceChunk]] = defaultdict(list)
        source_order: list[str] = []
        for chunk in ranked:
            if chunk.source_id not in groups:
                source_order.append(chunk.source_id)
            groups[chunk.source_id].append(chunk)

        out: list[EvidenceChunk] = []
        for round_no in range(max_per_source):
            added = False
            for source_id in source_order:
                group = groups[source_id]
                if round_no < len(group):
                    out.append(group[round_no])
                    added = True
                    if len(out) >= limit:
                        return out
            if not added:
                break
        return out

    def _expand_neighbors(
        self,
        seeds: list[EvidenceChunk],
        *,
        allowed_source_ids: set[str] | None,
        limit: int,
    ) -> list[EvidenceChunk]:
        out: list[EvidenceChunk] = []
        seen: set[str] = set()

        def add(chunk: EvidenceChunk, score: float | None = None):
            if chunk.chunk_id in seen or len(out) >= limit:
                return
            if not self._allowed(chunk.source_id, allowed_source_ids):
                return
            seen.add(chunk.chunk_id)
            if score is None:
                out.append(chunk)
            else:
                out.append(EvidenceChunk(**{**chunk.__dict__, "score": score}))

        for seed in seeds:
            add(seed)
        # Add adjacent context after all primary evidence so a neighbor cannot
        # crowd out a stronger seed from another source.
        for seed in seeds:
            group = self.by_source.get(seed.source_id) or []
            positions = {chunk.ordinal: chunk for chunk in group}
            for ordinal in (seed.ordinal - 1, seed.ordinal + 1):
                neighbor = positions.get(ordinal)
                if neighbor is not None:
                    add(neighbor, max(seed.score * 0.75, 0.0))
        return out[:limit]

    def search(
        self,
        query: str,
        *,
        allowed_source_ids: set[str] | None = None,
        candidate_k: int = 24,
        evidence_k: int = 10,
    ) -> list[EvidenceChunk]:
        if not query.strip() or allowed_source_ids == set() or evidence_k <= 0:
            return []

        overview = is_overview_query(query)
        global_overview = is_global_overview_query(query)
        candidate_k = max(candidate_k, evidence_k * 2)

        lexical = self.lexical.search(
            query,
            top_k=candidate_k,
            allowed_source_ids=allowed_source_ids,
        )
        semantic = (
            self.semantic.search(
                query,
                top_k=candidate_k,
                allowed_source_ids=allowed_source_ids,
            )
            if self.semantic else []
        )
        rankings = [lexical]
        if semantic:
            rankings.append(semantic)
        fused = reciprocal_rank_fusion(rankings, top_k=candidate_k)

        if global_overview:
            representatives = self._representative_chunks(
                allowed_source_ids,
                limit=max(evidence_k, 1),
            )
            combined: list[EvidenceChunk] = []
            seen: set[str] = set()
            for chunk in representatives + fused:
                if chunk.chunk_id not in seen:
                    combined.append(chunk)
                    seen.add(chunk.chunk_id)
            seeds = self._diversify(
                combined,
                limit=evidence_k,
                max_per_source=1,
            )
        elif overview:
            seeds = self._diversify(
                fused,
                limit=max(1, evidence_k - 2),
                max_per_source=2,
            )
            # If a broad question has weak lexical matches, keep the corpus visible
            # with a couple of source representatives instead of returning nothing.
            if len(seeds) < min(4, evidence_k):
                seen = {chunk.chunk_id for chunk in seeds}
                for chunk in self._representative_chunks(
                    allowed_source_ids,
                    limit=evidence_k,
                ):
                    if chunk.chunk_id not in seen:
                        seeds.append(chunk)
                        seen.add(chunk.chunk_id)
                    if len(seeds) >= min(4, evidence_k):
                        break
        else:
            seeds = self._diversify(
                fused,
                limit=max(1, evidence_k - 2),
                max_per_source=4,
            )

        if not seeds:
            return []

        return self._expand_neighbors(
            seeds,
            allowed_source_ids=allowed_source_ids,
            limit=evidence_k,
        )