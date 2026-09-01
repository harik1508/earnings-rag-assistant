"""
Streamlit dashboard: query the assistant + visualize logged eval/latency/cost
metrics. Run with: streamlit run frontend/dashboard.py

This is the piece that turns "I built a RAG system" into "I built a RAG
system and can show you it working, plus what it costs and how good it is" —
the second sentence is what an interviewer actually wants to hear.
"""

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"
LOG_PATH = Path("data/processed/query_log.jsonl")

st.set_page_config(page_title="Earnings Call Research Assistant", layout="wide")
st.title("Earnings Call / Investor Research Assistant")

tab_query, tab_observability = st.tabs(["Ask a question", "Observability dashboard"])

with tab_query:
    company = st.selectbox("Filter by company (optional)", ["Any", "MSFT", "NVDA", "JPM", "COST", "LUV"])
    question = st.text_input("Ask about a recent earnings call", placeholder="e.g. What did management say about margin pressure?")

    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating..."):
            payload = {"question": question, "top_k": 5}
            if company != "Any":
                payload["company"] = company
            try:
                resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.markdown("### Answer")
                st.write(data["answer"])
                st.caption(
                    f"Latency: {data['latency_ms']:.0f}ms | "
                    f"Est. cost: ${data['estimated_cost_usd']:.5f} | "
                    f"Cache hit: {data['cache_hit']} | "
                    f"Sources: {', '.join(data['retrieved_chunk_ids']) or 'none'}"
                )
                # sub_questions has >1 entry only when query decomposition
                # actually split the question into per-quarter sub-queries —
                # surfacing this makes the Step 6 finding (decomposition
                # fixes cross-quarter retrieval coverage) visible in the demo,
                # not just in the README.
                if len(data.get("sub_questions", [])) > 1:
                    with st.expander(f"Question was decomposed into {len(data['sub_questions'])} sub-queries"):
                        for sq in data["sub_questions"]:
                            st.write(f"- {sq}")
            except requests.exceptions.ConnectionError:
                st.error("API not running — start it with `uvicorn api.main:app --reload`")

with tab_observability:
    if not LOG_PATH.exists():
        st.info("No queries logged yet — ask some questions first.")
    else:
        rows = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries", len(df))
        col2.metric("Avg latency (ms)", f"{df['latency_ms'].mean():.0f}" if len(df) else "—")
        col3.metric("Total est. cost ($)", f"{df['estimated_cost_usd'].sum():.4f}" if len(df) else "—")

        st.markdown("### Query log")
        st.dataframe(df.sort_values("latency_ms", ascending=False), use_container_width=True)

        st.markdown("### Latency over time")
        if len(df) > 1:
            st.line_chart(df["latency_ms"])

    st.markdown("---")
    st.caption(
        "Eval scores (faithfulness/relevance/evasiveness_handling) from "
        "eval/llm_judge.py can be merged into this log once you run the eval "
        "suite — add them as extra columns here for the full picture."
    )