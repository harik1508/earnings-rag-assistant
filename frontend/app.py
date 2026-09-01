"""
Consolidated, deployable version of the assistant: everything runs
in-process inside this single Streamlit app, rather than Streamlit calling
a separate FastAPI server over HTTP (see api/main.py for that version,
kept in the repo as a demonstration of a proper service layer, but not
what's actually deployed here).

Why consolidate for deployment: running two coordinated services (a FastAPI
backend + a Streamlit frontend) adds real operational complexity for a
portfolio demo -- two hosts, CORS, cold-start timing between them, free
tiers that sleep independently. A single Streamlit app deployed on
Streamlit Community Cloud has none of that: one repo, one host, one URL.

Run locally with: streamlit run frontend/app.py
Deploy by pointing Streamlit Community Cloud at this file in your repo.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# set_page_config MUST be the literal first Streamlit command executed in
# the script -- even accessing st.secrets below (when no secrets.toml
# exists) renders an internal error banner INSIDE the app, which itself
# counts as a "command" and breaks this rule if it runs first. So this has
# to come before anything else that touches `st`.
st.set_page_config(page_title="Earnings Call Research Assistant", layout="wide")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))

# On Streamlit Community Cloud, secrets are exposed via st.secrets, not as
# real environment variables -- but the OpenAI SDK and embeddings/benchmark.py
# both read os.environ["OPENAI_API_KEY"] directly. Bridging st.secrets into
# os.environ here means the exact same code works unchanged both locally
# (where OPENAI_API_KEY is a real env var you set yourself) and on Streamlit
# Cloud (where it comes from the secrets manager) -- no separate code path
# needed for each environment.
#
# Only attempt this if the key ISN'T already in the environment. Locally,
# it already is (you set it with $env:), so this whole block is skipped --
# which also avoids the cosmetic "No secrets found" banner that merely
# ACCESSING st.secrets renders when no secrets.toml exists, independent of
# whether the resulting exception gets caught. On Streamlit Cloud, the key
# won't be in os.environ yet, so this runs and st.secrets will have it.
if "OPENAI_API_KEY" not in os.environ:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        pass  # no secrets.toml at all -- fine, nothing to bridge

from openai import OpenAI
from benchmark import embed_local
from hybrid_search import HybridRetriever
from query_decomposition import decomposed_retrieve

GENERATION_MODEL = "gpt-4o-mini"
CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks"
CHUNKS_PATTERN = "*_turns_overlap.json"  # winning chunking strategy from Step 4
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "query_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006

client = OpenAI()


@st.cache_resource(show_spinner="Building retriever (embedding chunks + BM25 index) — one-time cost...")
def get_retriever() -> HybridRetriever:
    """
    st.cache_resource is Streamlit's built-in tool for exactly this problem:
    build something expensive ONCE and reuse it across every user session
    and every rerun of the script (Streamlit reruns the whole file on every
    interaction), rather than rebuilding a BM25 index and re-embedding 800+
    chunks on every single query -- the same problem the FastAPI startup
    hook in api/main.py solves for that version of the app.
    """
    chunks = []
    for f in sorted(CHUNKS_DIR.glob(CHUNKS_PATTERN)):
        chunks.extend(json.loads(f.read_text()))
    if not chunks:
        st.error(f"No chunk files found matching {CHUNKS_PATTERN} in {CHUNKS_DIR}")
        st.stop()
    return HybridRetriever(chunks, embed_local)


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
    prompt = build_grounded_prompt(question, retrieved_chunks)
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS) + (output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)


def log_query(entry: dict):
    # NOTE: on Streamlit Community Cloud the filesystem is ephemeral -- this
    # log resets whenever the app restarts or redeploys. Fine for a live
    # demo's session-level observability; not a substitute for real
    # persistent logging (e.g. a hosted DB) in an actual production system.
    try:
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging failure should never break the user-facing answer


st.title("Earnings Call / Investor Research Assistant")
st.caption(
    "RAG system over MSFT, NVDA, JPM, COST, and LUV earnings call transcripts — "
    "hybrid retrieval (BM25 + vector) with automatic query decomposition for "
    "cross-quarter questions. [Project write-up](https://github.com) for the full "
    "eval methodology and findings."
)

tab_query, tab_observability = st.tabs(["Ask a question", "Observability dashboard"])

with tab_query:
    company = st.selectbox("Filter by company (optional)", ["Any", "MSFT", "NVDA", "JPM", "COST", "LUV"])
    question = st.text_input(
        "Ask about a recent earnings call",
        placeholder="e.g. What did management say about margin pressure?",
    )

    if st.button("Ask") and question:
        start = time.time()
        with st.spinner("Retrieving and generating..."):
            retriever = get_retriever()
            retrieved, sub_questions = decomposed_retrieve(
                retriever, question, top_k_per_subquery=5, decompose_model=GENERATION_MODEL
            )
            if company != "Any":
                retrieved = [c for c in retrieved if c.chunk_id.startswith(company)]

            answer, input_tokens, output_tokens = generate_answer(question, retrieved)
            latency_ms = (time.time() - start) * 1000
            cost = estimate_cost(input_tokens, output_tokens)

            log_query({
                "question": question, "company": company, "answer": answer,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
                "latency_ms": latency_ms, "estimated_cost_usd": cost,
            })

        st.markdown("### Answer")
        st.write(answer)
        st.caption(
            f"Latency: {latency_ms:.0f}ms | Est. cost: ${cost:.5f} | "
            f"Sources: {', '.join(c.chunk_id for c in retrieved) or 'none'}"
        )
        if len(sub_questions) > 1:
            with st.expander(f"Question was decomposed into {len(sub_questions)} sub-queries"):
                for sq in sub_questions:
                    st.write(f"- {sq}")

with tab_observability:
    if not LOG_PATH.exists():
        st.info("No queries logged yet in this session — ask something first.")
    else:
        rows = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries", len(df))
        col2.metric("Avg latency (ms)", f"{df['latency_ms'].mean():.0f}" if len(df) else "—")
        col3.metric("Total est. cost ($)", f"{df['estimated_cost_usd'].sum():.4f}" if len(df) else "—")

        st.markdown("### Query log")
        st.dataframe(df.sort_values("latency_ms", ascending=False), use_container_width=True)

    st.markdown("---")
    st.caption(
        "This log is session/instance-local and resets on app restart. "
        "See eval/eval_results.json in the repo for the full, persistent "
        "faithfulness/relevance/evasiveness-handling scores from the LLM-judge harness."
    )