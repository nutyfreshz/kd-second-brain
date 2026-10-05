from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Iterable
from functools import lru_cache
import threading

from core.schemas import EvidenceChunk


_TOKEN_RE = re.compile(r"(?:[A-Za-z][A-Za-z0-9]*|\d+[A-Za-z][A-Za-z0-9]*)(?:[-_/][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?|[\u0E00-\u0E7F]+", re.UNICODE)


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
        if not query.strip() or allowed_source_ids == set() or evidence_k <= 0:
            return []
        lexical = self.lexical.search(query, top_k=candidate_k, allowed_source_ids=allowed_source_ids)
        semantic = (
            self.semantic.search(query, top_k=candidate_k, allowed_source_ids=allowed_source_ids)
            if self.semantic else []
        )
        rankings = [lexical]
        if semantic:
            rankings.append(semantic)
        return reciprocal_rank_fusion(rankings, top_k=evidence_k)
