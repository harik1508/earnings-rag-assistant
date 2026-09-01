"""
FastAPI serving layer for the earnings call RAG assistant.

Every query logs: retrieved chunk IDs, latency, estimated token cost, and
(optionally, async) an eval score — this log is what feeds the observability
dashboard in frontend/dashboard.py. A RAG demo without this logging can
answer questions; this one can also tell you when and why it's wrong.

The retrieval/generation pipeline here is the exact same one validated in
eval/run_eval.py (hybrid search + query decomposition, no reranking — since
reranking measurably hurt Recall@10 in the Step 5 benchmark) — this file
just wraps it behind an HTTP endpoint rather than a batch eval loop.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from benchmark import embed_local
from hybrid_search import HybridRetriever
from query_decomposition import decomposed_retrieve

app = FastAPI(title="Earnings Call Research Assistant")

client = OpenAI()  # reads OPENAI_API_KEY from the environment
GENERATION_MODEL = "gpt-4o-mini"
CHUNKS_DIR = Path("data/processed/chunks")
CHUNKS_PATTERN = "*_turns_overlap.json"  # the winning chunking strategy from Step 4

LOG_PATH = Path("data/processed/query_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Simple in-memory cache keyed by a hash of the question. Swap for Redis if
# you deploy this beyond a single-process demo.
_cache: dict[str, dict] = {}

# The retriever holds a BM25 index and embeddings for every chunk -- built
# ONCE at first use, not per-request. Rebuilding this on every query would
# re-embed 800+ chunks and rebuild the BM25 index every single time, which
# is both slow (seconds, not milliseconds) and wastes API/compute cost for
# no benefit, since the underlying chunk corpus doesn't change between
# requests.
_retriever: HybridRetriever | None = None

# Actual current OpenAI pricing for gpt-4o-mini, per 1K tokens. Update this
# if you change models -- see requirements.txt / README for how pricing was
# looked up.
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006


class QueryRequest(BaseModel):
    question: str
    company: str | None = None  # optional filter, e.g. "MSFT"
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: list[str]
    sub_questions: list[str]
    latency_ms: float
    estimated_cost_usd: float
    cache_hit: bool


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        chunks = []
        for f in sorted(CHUNKS_DIR.glob(CHUNKS_PATTERN)):
            chunks.extend(json.loads(f.read_text()))
        if not chunks:
            raise RuntimeError(
                f"No chunk files found matching {CHUNKS_PATTERN} in {CHUNKS_DIR} -- "
                "run ingestion/chunking.py across your corpus first."
            )
        print(f"[startup] Building retriever: embedding {len(chunks)} chunks + building BM25 index "
              "(this happens once and can take 30-90s depending on hardware)...")
        _retriever = HybridRetriever(chunks, embed_local)
        print("[startup] Retriever ready.")
    return _retriever


@app.on_event("startup")
def warm_up_retriever():
    """
    Builds the retriever (BM25 index + embeddings for every chunk) at
    server startup, BEFORE uvicorn starts accepting requests -- not lazily
    on the first incoming request. Embedding 800+ chunks locally can easily
    take longer than a typical client-side HTTP timeout (e.g. the 30s
    default used by frontend/dashboard.py's requests.post call); if that
    work happened inside the first request handler instead, the very first
    user query would be the one paying that cost and risking a timeout --
    exactly what happened before this was moved here. Every request after
    startup, including the first one a user actually sends, hits an
    already-warm retriever and returns quickly.
    """
    get_retriever()


def _cache_key(question: str, company: str | None) -> str:
    return hashlib.sha256(f"{question}|{company}".encode()).hexdigest()


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS
        + output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS
    )


def build_grounded_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(f"[{c.chunk_id}]: {c.text}" for c in retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the source \
material shows management gave a hedged or non-committal answer, say so \
explicitly rather than inventing a confident specific answer. Cite which \
chunk(s) support your answer.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list) -> tuple[str, int, int]:
    """Returns (answer_text, input_tokens, output_tokens) -- real usage
    figures from the API response, not estimates, so cost tracking here is
    exact rather than approximate."""
    prompt = build_grounded_prompt(question, retrieved_chunks)
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content
    return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens


def log_query(entry: dict):
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()
    cache_key = _cache_key(req.question, req.company)

    if cache_key in _cache:
        cached = _cache[cache_key]
        latency_ms = (time.time() - start) * 1000
        log_query({**cached, "cache_hit": True, "latency_ms": latency_ms})
        return QueryResponse(**cached, latency_ms=latency_ms, cache_hit=True)

    retriever = get_retriever()

    # decomposed_retrieve is a strict generalization of a plain search: for
    # single-period questions it's a no-op (returns the question unchanged),
    # only multi-quarter questions actually get split. Same logic validated
    # in eval/run_eval.py, applied here to live traffic.
    retrieved, sub_questions = decomposed_retrieve(
        retriever, req.question, top_k_per_subquery=req.top_k, decompose_model=GENERATION_MODEL
    )

    # Optional company filter: applied AFTER retrieval rather than by
    # building a separate per-company index. Chunk IDs are prefixed with
    # the company ticker (e.g. "MSFT_2026_Q4_turnoverlap_19"), so this is a
    # cheap string-prefix filter -- simpler than maintaining N separate
    # retriever indices for a project this size, at the cost of occasionally
    # retrieving slightly fewer than top_k chunks if the company filter
    # removes some of the best matches. A production system with heavier
    # traffic would likely build per-company indices instead.
    if req.company:
        retrieved = [c for c in retrieved if c.chunk_id.startswith(req.company.upper())]

    answer, input_tokens, output_tokens = generate_answer(req.question, retrieved)

    latency_ms = (time.time() - start) * 1000
    cost = estimate_cost(input_tokens, output_tokens)

    result = {
        "answer": answer,
        "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
        "sub_questions": sub_questions,
        "estimated_cost_usd": cost,
    }
    _cache[cache_key] = result
    log_query({
        **result,
        "question": req.question,
        "company": req.company,
        "cache_hit": False,
        "latency_ms": latency_ms,
    })

    return QueryResponse(**result, latency_ms=latency_ms, cache_hit=False)


@app.get("/health")
def health():
    return {"status": "ok", "retriever_loaded": _retriever is not None}