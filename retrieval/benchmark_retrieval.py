"""
Benchmarks retrieval quality across three configurations, reusing the same
gold Q&A set, chunk pool, and content-overlap matching logic as
embeddings/benchmark.py so the numbers are directly comparable:

  1. vector-only   — pure cosine similarity (this is what embeddings/benchmark.py
                      already measured: speaker_turn_overlap = 62.5% / 68.8%)
  2. hybrid         — BM25 (lexical) + vector, merged, no reranking
  3. hybrid+rerank  — same, then reordered by a cross-encoder

Why this matters for earnings calls specifically: analysts and executives
use exact, distinctive terms — "free cash flow", "net interest margin",
specific dollar figures, product names like "VeraRubin" — that a small
local embedding model doesn't always weight as heavily as a human would.
BM25 is a literal keyword-overlap score, so it catches exact-term matches
that cosine similarity alone can under-rank. Reranking then re-scores the
merged candidate list with a model trained specifically to judge
query-passage relevance, which is a more precise (if slower) signal than
either retrieval method alone.

Usage:
    python retrieval/benchmark_retrieval.py \\
        --gold_qa eval/gold_qa.json \\
        --chunks_dir data/processed/chunks/ \\
        --pattern "*_turns_overlap.json"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import embed_local, load_reference_texts, get_gold_ids, contains_reference
from hybrid_search import HybridRetriever


def load_chunks_from_dir(chunks_dir: Path, pattern: str) -> list[dict]:
    chunks = []
    files = sorted(chunks_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {chunks_dir}")
    for f in files:
        chunks.extend(json.loads(f.read_text()))
    print(f"Loaded {len(chunks)} chunks from {len(files)} files matching '{pattern}'")
    return chunks


def evaluate_retriever_mode(
    gold_qa: list[dict],
    retriever: HybridRetriever,
    reference_texts: dict[str, str],
    rerank: bool,
    label: str,
) -> dict:
    hits_at_5, hits_at_10, skipped = 0, 0, 0

    for qa in gold_qa:
        gold_ids = get_gold_ids(qa)
        gold_texts = [reference_texts[gid] for gid in gold_ids if gid in reference_texts]
        if not gold_texts:
            skipped += 1
            continue

        results = retriever.search(qa["question"], top_k=10, rerank=rerank)
        top5_texts = [r.text for r in results[:5]]
        top10_texts = [r.text for r in results[:10]]

        if any(contains_reference(c, ref) for c in top5_texts for ref in gold_texts):
            hits_at_5 += 1
        if any(contains_reference(c, ref) for c in top10_texts for ref in gold_texts):
            hits_at_10 += 1

    n = len(gold_qa) - skipped
    return {
        "label": label,
        "recall_at_5": hits_at_5 / n if n else 0.0,
        "recall_at_10": hits_at_10 / n if n else 0.0,
        "n": n,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_qa", required=True)
    parser.add_argument("--chunks_dir", required=True)
    parser.add_argument("--pattern", default="*_turns_overlap.json",
                         help="Which chunk strategy to search over (defaults to the winning strategy from Step 4)")
    parser.add_argument("--bm25_weight", type=float, default=0.4)
    args = parser.parse_args()

    gold_qa = json.loads(Path(args.gold_qa).read_text())
    chunks = load_chunks_from_dir(Path(args.chunks_dir), args.pattern)
    reference_texts = load_reference_texts(Path(args.chunks_dir))

    print("Building BM25 index and embedding all chunks (one-time cost for this run)...")
    retriever = HybridRetriever(chunks, embed_local)

    results = []
    print("Evaluating hybrid (no rerank)...")
    results.append(evaluate_retriever_mode(gold_qa, retriever, reference_texts, rerank=False, label="hybrid (BM25 + vector, no rerank)"))
    print("Evaluating hybrid + rerank (this loads a cross-encoder model, may take a moment)...")
    results.append(evaluate_retriever_mode(gold_qa, retriever, reference_texts, rerank=True, label="hybrid + reranked"))

    print(f"\n{'Config':<38}{'Recall@5':<12}{'Recall@10':<12}{'N (skipped)':<15}")
    print(f"{'vector-only (from embeddings/benchmark.py)':<38}{'—':<12}{'—':<12}{'see prior run':<15}")
    for r in results:
        print(f"{r['label']:<38}{r['recall_at_5']:<12.3f}{r['recall_at_10']:<12.3f}{r['n']} ({r['skipped']} skipped)")


if __name__ == "__main__":
    main()
