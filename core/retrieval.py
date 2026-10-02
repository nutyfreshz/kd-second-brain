from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable

from core.schemas import EvidenceChunk


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?|[\u0E00-\u0E7F]+", re.UNICODE)


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
        for doc in self.docs:
            counts = Counter(doc)
            self.tf.append(counts)
            self.df.update(counts.keys())

    def search(self, query: str, top_k: int = 12) -> list[tuple[EvidenceChunk, float]]:
        if not self.chunks:
            return []
        q = tokenize(query)
        n = len(self.chunks)
        scored: list[tuple[EvidenceChunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            dl = self.lengths[i] or 1
            for term in q:
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1) / denom)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


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
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            passages = [f"passage: {c.title}\n{c.heading}\n{c.text}" for c in self.chunks]
            self._vectors = self._model.encode(
                passages, normalize_embeddings=True, show_progress_bar=False
            )
            self.available = True
        except Exception as exc:
            self.error = type(exc).__name__

    def search(self, query: str, top_k: int = 12) -> list[tuple[EvidenceChunk, float]]:
        if not self.available or self._model is None or self._vectors is None:
            return []
        import numpy as np
        q = self._model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        scores = np.dot(self._vectors, q)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in indices]


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

    @property
    def limited_mode(self) -> bool:
        return self.semantic is None or not self.semantic.available

    def search(
        self,
        query: str,
        *,
        allowed_source_ids: set[str] | None = None,
        candidate_k: int = 12,
        evidence_k: int = 6,
    ) -> list[EvidenceChunk]:
        lexical = self.lexical.search(query, top_k=max(candidate_k * 2, candidate_k))
        semantic = (
            self.semantic.search(query, top_k=max(candidate_k * 2, candidate_k))
            if self.semantic else []
        )
        if allowed_source_ids is not None:
            lexical = [(c, s) for c, s in lexical if c.source_id in allowed_source_ids]
            semantic = [(c, s) for c, s in semantic if c.source_id in allowed_source_ids]
        rankings = [lexical]
        if semantic:
            rankings.append(semantic)
        return reciprocal_rank_fusion(rankings, top_k=evidence_k)
