"""
Hybrid retrieval: combines BM25 (lexical/keyword match) with dense vector
search, then optionally reranks the merged candidates with a cross-encoder.

Why hybrid instead of vector-only:
Earnings calls are full of exact numbers, tickers, and named metrics
("free cash flow", "data center revenue", "net interest margin"). Pure dense
retrieval sometimes under-ranks a chunk that contains the exact right number
but is semantically "distant" in embedding space from how the question was
phrased. BM25 catches these exact-match cases; vector search catches
paraphrased/conceptual questions. Combining both is a well-established
pattern for reducing retrieval misses in this kind of factual+conversational
corpus — this file is where you produce the "+ hybrid search" row in the
README results table.
"""

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import CrossEncoder
    _reranker = None
except ImportError:
    _reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str  # "bm25" | "vector" | "hybrid" | "reranked"


class HybridRetriever:
    def __init__(self, chunks: list[dict], embed_fn):
        """
        chunks: list of {"chunk_id": ..., "text": ...}
        embed_fn: callable(list[str]) -> np.ndarray, e.g. embeddings.benchmark.embed_local
        """
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]

        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        self.embed_fn = embed_fn
        self.doc_vecs = embed_fn(self.texts)

    def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_vec = self.embed_fn([query])[0]
        q_norm = q_vec / np.linalg.norm(q_vec)
        d_norm = self.doc_vecs / np.linalg.norm(self.doc_vecs, axis=1, keepdims=True)
        sims = d_norm @ q_norm
        ranked = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in ranked]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    @staticmethod
    def _normalize(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return scores
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def search(
        self, query: str, top_k: int = 10, candidate_pool: int = 30,
        rerank: bool = True, bm25_weight: float = 0.4,
    ) -> list[RetrievedChunk]:
        vector_hits = dict(self._vector_search(query, candidate_pool))
        bm25_hits = dict(self._bm25_search(query, candidate_pool))

        vector_norm = self._normalize(vector_hits)
        bm25_norm = self._normalize(bm25_hits)

        combined_idx = set(vector_hits) | set(bm25_hits)
        combined_scores = {
            i: bm25_weight * bm25_norm.get(i, 0.0) + (1 - bm25_weight) * vector_norm.get(i, 0.0)
            for i in combined_idx
        }
        ranked = sorted(combined_scores.items(), key=lambda x: -x[1])[:candidate_pool]

        candidates = [
            RetrievedChunk(self.chunk_ids[i], self.texts[i], score, "hybrid")
            for i, score in ranked
        ]

        if rerank and candidates:
            reranker = get_reranker()
            pairs = [[query, c.text] for c in candidates]
            rerank_scores = reranker.predict(pairs)
            for c, s in zip(candidates, rerank_scores):
                c.score = float(s)
                c.source = "reranked"
            candidates.sort(key=lambda c: -c.score)

        return candidates[:top_k]
