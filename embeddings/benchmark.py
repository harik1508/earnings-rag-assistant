"""
Benchmarks two embedding models against each other on retrieval quality,
rather than defaulting to whichever one a tutorial used.

Compares:
  - OpenAI text-embedding-3-small (API, costs money, strong general quality)
  - BAAI/bge-small-en-v1.5 (local, free, no API dependency)

The metric here is a simple proxy: for each gold Q&A pair, does the correct
source chunk appear in the top-k retrieved results (Recall@k)? This is
distinct from the full LLM-as-judge eval in eval/llm_judge.py, which grades
the *generated answer* — this benchmark grades *retrieval* in isolation, so
you can tell whether a bad answer is a retrieval problem or a generation
problem. That separation is itself worth calling out in interviews.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
    _openai_client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    _openai_client = None

_local_model = None


def get_local_model():
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _local_model


def embed_openai(texts: list[str]) -> np.ndarray:
    if _openai_client is None:
        raise RuntimeError("OPENAI_API_KEY not set — skipping OpenAI embedding benchmark")
    resp = _openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


def embed_local(texts: list[str]) -> np.ndarray:
    model = get_local_model()
    return model.encode(texts, normalize_embeddings=True)


def cosine_sim_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q_norm = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    d_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return q_norm @ d_norm.T


@dataclass
class RecallResult:
    model_name: str
    recall_at_5: float
    recall_at_10: float
    n_queries: int
    skipped: int  # gold entries whose reference text couldn't be resolved at all


REFERENCE_PATTERN = "*_turns.json"  # gold_chunk_id(s) always point into THIS
                                     # strategy's files, regardless of which
                                     # strategy is currently being benchmarked
                                     # (see load_reference_texts below)


def get_gold_ids(qa: dict) -> list[str]:
    if qa.get("gold_chunk_ids"):
        return list(qa["gold_chunk_ids"])
    if qa.get("gold_chunk_id"):
        return [qa["gold_chunk_id"]]
    return []


def load_reference_texts(chunks_dir: Path) -> dict[str, str]:
    """
    gold_chunk_id / gold_chunk_ids values were resolved against the
    speaker_turn strategy's chunk_ids specifically (that's how gold_qa.json
    was built) — they are NOT valid ids in a different strategy's output
    (e.g. fixed_size chunk_ids follow a completely different scheme covering
    different text spans). So this reference lookup always loads the
    *_turns.json files, independent of which strategy's chunks are actually
    being searched over in a given benchmark run. This is what makes an
    apples-to-apples comparison between strategies possible at all.
    """
    ref = {}
    for f in sorted(chunks_dir.glob(REFERENCE_PATTERN)):
        for c in json.loads(f.read_text()):
            ref[c["chunk_id"]] = c["text"]
    return ref


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_reference(candidate_text: str, reference_text: str, window: int = 60, stride: int = 30) -> bool:
    """
    True if a meaningful, contiguous slice of `reference_text` appears
    verbatim inside `candidate_text`. This is the strategy-agnostic
    replacement for exact chunk_id equality: since fixed_size and
    speaker_turn chunking both slice the SAME underlying transcript text,
    a chunk from either strategy that genuinely covers the answer will
    contain a real substring of the reference text, even though the two
    strategies' chunk boundaries (and therefore chunk_ids) never line up.
    Slides a window across the reference text rather than requiring an
    exact full-text match, since the two strategies' chunks are unlikely to
    have IDENTICAL boundaries — only overlapping coverage.
    """
    candidate_norm = normalize(candidate_text)
    reference_norm = normalize(reference_text)
    if len(reference_norm) <= window:
        return reference_norm in candidate_norm
    for start in range(0, len(reference_norm) - window, stride):
        if reference_norm[start:start + window] in candidate_norm:
            return True
    return False


def evaluate_recall(
    gold_qa: list[dict],
    chunks: list[dict],       # the candidate pool being searched (either strategy)
    reference_texts: dict[str, str],  # chunk_id -> text, ALWAYS from the turns strategy
    embed_fn,
    model_name: str,
) -> RecallResult:
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]
    doc_vecs = embed_fn(chunk_texts)

    questions = [qa["question"] for qa in gold_qa]
    query_vecs = embed_fn(questions)

    sims = cosine_sim_matrix(query_vecs, doc_vecs)
    hits_at_5, hits_at_10, skipped = 0, 0, 0

    for i, qa in enumerate(gold_qa):
        gold_ids = get_gold_ids(qa)
        gold_texts = [reference_texts[gid] for gid in gold_ids if gid in reference_texts]
        if not gold_texts:
            skipped += 1
            continue

        ranked_idx = np.argsort(-sims[i])
        top5_texts = [chunk_texts[j] for j in ranked_idx[:5]]
        top10_texts = [chunk_texts[j] for j in ranked_idx[:10]]

        hit5 = any(contains_reference(c, ref) for c in top5_texts for ref in gold_texts)
        hit10 = any(contains_reference(c, ref) for c in top10_texts for ref in gold_texts)
        if hit5:
            hits_at_5 += 1
        if hit10:
            hits_at_10 += 1

    n = len(gold_qa) - skipped
    return RecallResult(
        model_name=model_name,
        recall_at_5=hits_at_5 / n if n else 0.0,
        recall_at_10=hits_at_10 / n if n else 0.0,
        n_queries=n,
        skipped=skipped,
    )


def load_chunks_from_dir(chunks_dir: Path, pattern: str) -> list[dict]:
    """
    Loads and concatenates every chunk file matching `pattern` in
    chunks_dir into one pool. Cross-quarter gold questions need chunks from
    multiple quarters' files present simultaneously, so the benchmark
    always operates over a directory of files, not a single quarter.
    """
    chunks = []
    files = sorted(chunks_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {chunks_dir}")
    for f in files:
        chunks.extend(json.loads(f.read_text()))
    print(f"Loaded {len(chunks)} chunks from {len(files)} files matching '{pattern}'")
    return chunks


def run_benchmark(gold_qa_path: Path, chunks_dir: Path, pattern: str):
    gold_qa = json.loads(gold_qa_path.read_text())
    chunks = load_chunks_from_dir(chunks_dir, pattern)
    reference_texts = load_reference_texts(chunks_dir)

    results = [evaluate_recall(gold_qa, chunks, reference_texts, embed_local, "bge-small-en-v1.5 (local)")]

    if _openai_client is not None:
        results.append(evaluate_recall(gold_qa, chunks, reference_texts, embed_openai, "text-embedding-3-small (OpenAI)"))
    else:
        print("Skipping OpenAI benchmark — set OPENAI_API_KEY to include it.")

    print(f"\n{'Model':<40}{'Recall@5':<12}{'Recall@10':<12}{'N (skipped)':<15}")
    for r in results:
        print(f"{r.model_name:<40}{r.recall_at_5:<12.3f}{r.recall_at_10:<12.3f}{r.n_queries} ({r.skipped} skipped)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_qa", required=True, help="Path to eval/gold_qa.json")
    parser.add_argument("--chunks_dir", required=True, help="Directory containing chunk files, e.g. data/processed/chunks/")
    parser.add_argument("--pattern", default="*_turns.json", help="Glob pattern for which chunk files to load, e.g. '*_turns.json' or '*_fixed.json'")
    args = parser.parse_args()
    run_benchmark(Path(args.gold_qa), Path(args.chunks_dir), args.pattern)
