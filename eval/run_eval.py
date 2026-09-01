"""
The full pipeline, tied together: for each gold question, retrieve context
with the winning Step 5 config (hybrid search, no reranking — reranking
measurably hurt Recall@10 in both benchmark runs), generate a grounded
answer with an LLM, then score that answer with the LLM-judge harness
against the retrieved context and the gold reference answer.

This is a genuinely different test than anything run so far. The Step 4/5
benchmarks only asked "did we retrieve the right chunk" (Recall@k) — they
never checked whether an LLM could actually produce a correct, grounded
answer FROM that chunk. A chunk can be technically relevant and still
produce a bad answer if it's fragmented, ambiguous, or if the LLM
misreads a hedged statement as a confident one. This script is what
actually tests that.

Usage:
    python eval/run_eval.py \\
        --gold_qa eval/gold_qa.json \\
        --chunks_dir data/processed/chunks/ \\
        --pattern "*_turns_overlap.json" \\
        --top_k 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import embed_local
from hybrid_search import HybridRetriever
from query_decomposition import decomposed_retrieve
from llm_judge import judge_answer, JudgeScore

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment

GENERATION_MODEL = "gpt-4o-mini"


def load_chunks_from_dir(chunks_dir: Path, pattern: str) -> list[dict]:
    chunks = []
    for f in sorted(chunks_dir.glob(pattern)):
        chunks.extend(json.loads(f.read_text()))
    return chunks


def build_grounded_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Mirrors api/main.py's build_grounded_prompt, duplicated here rather than
    imported so this eval script has no dependency on the FastAPI app
    (which needs a running server, request/response models, etc.) — this
    script needs to run standalone from the command line.
    """
    context = "\n\n".join(f"[{c.chunk_id}]: {c.text}" for c in retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the source \
material shows management gave a hedged or non-committal answer, say so \
explicitly rather than inventing a confident specific answer. Cite which \
chunk(s) support your answer.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list) -> str:
    prompt = build_grounded_prompt(question, retrieved_chunks)
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def run_full_eval(gold_qa_path: Path, chunks_dir: Path, pattern: str, top_k: int):
    gold_qa = json.loads(gold_qa_path.read_text())
    chunks = load_chunks_from_dir(chunks_dir, pattern)
    print(f"Loaded {len(chunks)} chunks, {len(gold_qa)} gold questions")

    print("Building retriever (BM25 index + embeddings)...")
    retriever = HybridRetriever(chunks, embed_local)

    results = []
    for i, qa in enumerate(gold_qa):
        print(f"[{i+1}/{len(gold_qa)}] {qa['id']}...")

        # decomposed_retrieve is a strict generalization of a plain search:
        # for single-period questions it decomposes to [question] unchanged
        # and behaves identically to retriever.search(). Only cross-quarter
        # questions actually get split into per-quarter sub-queries, which
        # is what fixes the "one quarter's chunk crowds out the others" bug
        # found in the Step 6 results.
        retrieved, sub_questions = decomposed_retrieve(
            retriever, qa["question"], top_k_per_subquery=top_k, decompose_model=GENERATION_MODEL
        )
        answer = generate_answer(qa["question"], retrieved)
        context_text = "\n\n".join(c.text for c in retrieved)

        score = judge_answer(
            question=qa["question"],
            retrieved_context=context_text,
            generated_answer=answer,
            gold_answer=qa.get("gold_answer"),
            question_id=qa["id"],
            model=GENERATION_MODEL,
        )

        results.append({
            "id": qa["id"],
            "company": qa["company"],
            "question_type": qa["question_type"],
            "question": qa["question"],
            "sub_questions": sub_questions,  # [] means no decomposition happened
            "gold_answer": qa.get("gold_answer"),
            "generated_answer": answer,
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "evasiveness_handling": score.evasiveness_handling,
            "judge_reasoning": score.reasoning,
        })

    return results


def print_summary(results: list[dict]):
    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_rel = sum(r["relevance"] for r in results) / n
    avg_evasive = sum(r["evasiveness_handling"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"OVERALL (n={n})")
    print(f"{'='*60}")
    print(f"Faithfulness:          {avg_faith:.2f} / 5")
    print(f"Relevance:             {avg_rel:.2f} / 5")
    print(f"Evasiveness handling:  {avg_evasive:.2f} / 5")

    # Break down by question_type -- this is the more interesting cut, since
    # evasiveness_test questions are the project's actual differentiator
    print(f"\n{'By question type':<20}{'N':<5}{'Faithfulness':<15}{'Relevance':<12}{'Evasiveness':<12}")
    for qtype in ["direct_factual", "evasiveness_test", "cross_quarter"]:
        subset = [r for r in results if r["question_type"] == qtype]
        if not subset:
            continue
        n_sub = len(subset)
        f = sum(r["faithfulness"] for r in subset) / n_sub
        rel = sum(r["relevance"] for r in subset) / n_sub
        ev = sum(r["evasiveness_handling"] for r in subset) / n_sub
        print(f"{qtype:<20}{n_sub:<5}{f:<15.2f}{rel:<12.2f}{ev:<12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_qa", required=True)
    parser.add_argument("--chunks_dir", required=True)
    parser.add_argument("--pattern", default="*_turns_overlap.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="eval/eval_results.json")
    args = parser.parse_args()

    results = run_full_eval(Path(args.gold_qa), Path(args.chunks_dir), args.pattern, args.top_k)
    print_summary(results)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {args.output}")
