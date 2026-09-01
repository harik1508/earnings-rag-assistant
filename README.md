# Earnings Call / Investor Research Assistant

A production-style Retrieval-Augmented Generation (RAG) system over quarterly
earnings call transcripts, built with a rigorous evaluation harness rather than
a single "does it answer questions" demo.

**Repo:** github.com/harik1508/earnings-rag-assistant
**Live demo:** _add your Streamlit Community Cloud URL here once deployed_

**Companies covered (v1):** Microsoft (MSFT), NVIDIA (NVDA), JPMorgan Chase (JPM),
Costco (COST), Southwest Airlines (LUV) — chosen deliberately to span sectors
with very different transcript language (growth-narrative tech, regulated
banking, thin-margin retail, operationally volatile airline).

## Why this project exists

Most RAG portfolio projects stop at "it retrieves and answers." This one exists
to demonstrate three things AI Engineer interviews actually probe:

1. **Retrieval quality is measurable, not assumed.** We benchmark chunking
   strategies and embedding models against each other instead of picking one
   by default.
2. **LLM answers need grounding checks.** Earnings calls are full of hedged,
   evasive, or non-committal answers from executives — a system that can't
   tell "management confirmed X" apart from "management deflected the
   question" will confidently hallucinate. We test for this directly.
3. **Production systems need observability.** Every query logs its retrieved
   chunks, latency, token cost, and an automated quality score — not just the
   final answer.

## Architecture

```
Earnings call transcripts (Motley Fool, plain text)
      │
      ▼
[ingestion/]   → parse, clean, speaker-tag (exec vs analyst vs operator),
                 handle 7+ distinct real-world transcript-format quirks
      │
      ▼
[ingestion/]   → chunk 3 ways: fixed_size, speaker_turn, speaker_turn_overlap
      │
      ▼
[embeddings/]  → benchmark chunking strategies via Recall@k (BGE-small)
      │
      ▼
[retrieval/]   → hybrid search (BM25 + vector) — benchmarked WITH and
                 WITHOUT cross-encoder reranking; reranking measurably hurt
                 Recall@10 in testing, so the deployed pipeline uses hybrid
                 search alone (see Results)
      │
      ▼
[retrieval/]   → query decomposition: multi-quarter questions are split into
                 one retrieval sub-query per quarter before searching, so no
                 single quarter's chunk gets crowded out of a shared top-k
      │
      ▼
[api/ or       → construct grounded prompt → generate (GPT-4o-mini) →
 frontend/]      log latency/cost/sources
      │
      ▼
[eval/]        → LLM-as-judge scoring (faithfulness, relevance, evasiveness
                 detection) against a 32-question hand-labeled gold set
```

Two serving options are in the repo:
- **`api/main.py`** — a proper FastAPI service layer (retriever built once
  at startup, not per-request) paired with `frontend/dashboard.py` calling
  it over HTTP. Kept as a demonstration of standard service architecture.
- **`frontend/app.py`** — the same pipeline consolidated into a single
  Streamlit app (retrieval + generation run in-process, cached via
  `st.cache_resource`). This is what's actually deployed, since running two
  coordinated services adds real operational complexity a single-service
  portfolio demo doesn't need.

## Status

- [x] Project scaffolding
- [x] Transcript ingestion pipeline (5 companies, 15 quarters, 7 distinct
      real-world parsing bugs found and fixed)
- [x] Chunking strategy comparison (3 strategies: fixed_size, speaker_turn,
      speaker_turn_overlap)
- [x] Embedding benchmark — BGE-small, Recall@5/@10 across all 3 chunking
      strategies (see Results below)
- [x] Gold eval set (32 hand-written Q&A pairs across all 5 companies: direct
      factual, evasiveness tests, cross-quarter)
- [x] Hybrid retrieval + reranking (BM25 + vector, cross-encoder reranking;
      see Results below)
- [x] LLM-as-judge eval harness — full pipeline (retrieval + generation +
      judging) run end-to-end across all 32 gold questions; real gaps
      identified and iterated on (cross-quarter retrieval coverage fixed via
      query decomposition, judge blind spot on omission documented,
      self-grading bias documented, one known remaining limitation
      documented rather than overfit-fixed) — see Results below
- [x] FastAPI serving layer with cost/latency logging — hybrid retrieval +
      query decomposition + generation wired end-to-end, retriever built
      once at server startup (not per-request) to avoid the first live
      query timing out on a cold BM25/embedding build
- [x] Frontend (Streamlit) with eval dashboard — live query + observability
      tabs, decomposition sub-queries surfaced in the UI
- [x] Deployment

## Results

Benchmarked against a 32-question gold set spanning all 5 companies (MSFT,
NVDA, JPM, COST, LUV) — direct-factual, evasiveness-test, and cross-quarter
question types.

| Config | Recall@5 | Recall@10 | Faithfulness | Relevance | Evasiveness Detection |
|---|---|---|---|---|---|
| speaker_turn chunking, no overlap (BGE-small) | 0.531 | 0.625 | — | — | — |
| fixed_size chunking, 50-token overlap (BGE-small) | 0.625 | 0.688 | — | — | — |
| speaker_turn chunking, 50-token overlap (BGE-small) | 0.719 | 0.781 | — | — | — |
| hybrid search (BM25 + vector), speaker_turn_overlap chunks | 0.750 | 0.875 | — | — | — |
| hybrid + cross-encoder reranking | 0.750 | 0.812 | — | — | — |
| **Full pipeline (hybrid retrieval + query decomposition, top_k=5, GPT-4o-mini generation)** | — | — | **4.81 / 5** | **4.97 / 5** | **5.00 / 5** |

**Finding (Step 4 — chunking/embedding benchmark):** the first comparison
(16-question gold set, 2 companies) showed naive fixed-size chunking (with
overlap) beating speaker-turn-aware chunking (without overlap) on
Recall@k — the opposite of the initial hypothesis. Isolating the variable
(adding the same 50-token overlap window to speaker_turn chunking, changing
nothing else) reversed the result: speaker_turn_overlap beat fixed_size at
every gold-set size tested. This confirms the original hypothesis was
directionally correct, and the first benchmark run was confounded by an
unequal overlap setting between the two strategies, not a flaw in the
turn-aware approach itself.

**Finding (Step 5 — hybrid retrieval benchmark):** adding BM25 (lexical
search) alongside vector search improved Recall@10 by ~9-19 points across
both gold-set sizes tested — the largest single improvement in the
project, and consistent with the domain hypothesis that earnings calls are
full of exact figures and named entities a small embedding model
under-weights. Cross-encoder reranking on top of hybrid search, however,
*reduced* Recall@10 in both the 16-question run (87.5%→81.2%, 1 flipped
answer) and the 32-question run (87.5%→81.2%, 2 flipped answers) — the same
proportional regression held after doubling the gold set's size and
company diversity, which is evidence this is a real effect (a
general-purpose reranker trained on MS MARCO passage relevance not
transferring cleanly to financial/earnings-call language), not sampling
noise from a small gold set. "Add reranking" is not a free win here without
measuring it — a useful, if unglamorous, finding to have actually tested
rather than assumed.

**Finding (Step 6 — full pipeline: generation + LLM-judge eval):** wiring
retrieval through to generation (GPT-4o-mini, hybrid search, top_k=5) and
scoring with an LLM judge produced high overall scores (faithfulness 4.78,
relevance 4.91, evasiveness handling 5.00), but a closer read of individual
results surfaces real, specific problems the aggregate numbers hide:

- *A perfect 5.00/5 on evasiveness handling across all 11 evasiveness-test
  questions is a red flag, not a clean win.* Generation and judging both
  used GPT-4o-mini — the same model grading its own output family. This is
  a known failure mode in LLM eval (self-preference bias): a model tends to
  rate outputs in its own style more favorably than an independent judge
  would. These scores should be read as "the system produced GPT-4o-mini-
  approved answers," not as an unbiased measurement. A stronger version of
  this harness would judge with a different model (e.g. Claude) than the
  one used to generate.
- *Cross-quarter faithfulness (4.00) is the one score low enough to trust
  as a real signal, and inspecting the actual transcripts explains why.*
  `msft_cross_quarter_01` needed chunks from Q2, Q3, AND Q4 to answer
  correctly — but a single top-5 retrieval call only surfaced Q1-Q3 chunks,
  never Q4. The generated answer correctly avoided fabricating the missing
  Q4 figure, but incorrectly claimed no data was available at all, when
  Q2/Q3 data it *did* have went unused. This is a retrieval-coverage
  problem, not a generation-quality problem: one query vector structurally
  can't reliably represent "give me data from four different quarters" at
  once.
- *The judge does not penalize omission, only fabrication.*
  `msft_cross_quarter_02` scored a perfect 5/5 faithfulness despite its
  retrieved context missing Q1's $392B RPO figure entirely and its answer
  never mentioning Q1 at all — because everything the answer DID say was
  accurate relative to what was retrieved. The current judge prompt
  (eval/llm_judge.py) only checks "is every claim grounded," not "is
  anything required missing." These are different properties.
  `nvda_cross_quarter_01`, by contrast, DID get caught (faithfulness 3) for
  a genuine hallucination — stating NVIDIA's Q1 FY2027 revenue as "$6.4
  billion" against an actual figure of $82 billion — confirming the judge
  isn't purely rubber-stamping everything, just that it has a specific,
  identifiable blind spot around completeness.
- *The judge model itself produced at least one incoherent scoring
  rationale.* On `jpm_cross_quarter_01`, the judge's reasoning field reads:
  "inaccurately states...increased from $95 billion to $96.5 billion, when
  it actually increased from $95 billion to $96.5 billion, which is
  correct" — self-contradictory, and a visible symptom of using a small,
  cheap model (GPT-4o-mini) as judge rather than a stronger one.

**Fix implemented — query decomposition (retrieval/query_decomposition.py):**
detect multi-quarter questions and issue one retrieval sub-query per quarter
(rather than one query expected to cover all of them), guaranteeing each
quarter gets its own retrieval slots instead of competing in a single shared
ranked list. First implementation caused a regression — the decomposition
LLM over-triggered on single-quarter questions that simply asked about
*multiple facts* (e.g. "revenue AND growth rate," "net income AND ROTCE"),
conflating "multiple facts" with "multiple time periods." This corrupted
`direct_factual` faithfulness (4.93→4.73) by splitting well-anchored
single-quarter questions into under-specified fragments — `nvda_2027_q1_01`
regressed from the correct $75B/92% to a hallucinated $68B/75% pulled from
the wrong quarter entirely.

Fixed with two layers: (1) added explicit negative examples to the
decomposition prompt showing single-quarter, multi-fact questions that must
NOT be split, and (2) a structural safeguard independent of the prompt —
after decomposition, count how many *distinct quarters* are actually
referenced across the sub-questions; if fewer than 2, discard the split and
fall back to the original question regardless of what the model returned.
The second layer matters more than the first: prompt instructions alone had
already been shown to fail once, so correctness needed to not depend on the
model reliably following instructions.

Result: `direct_factual` faithfulness fully recovered to 4.93 (matching the
pre-regression baseline), and `cross_quarter` faithfulness improved from the
original 4.00 to 4.17, with 5 of 6 cross-quarter questions now scoring
5/5/5 (MSFT Copilot seats, JPM full-year outlook, both COST comparisons,
and one further MSFT metric).

**Known remaining limitation:** `nvda_cross_quarter_01` (NVIDIA total
revenue + platform visibility, Q3 FY2026 → Q1 FY2027) still scores poorly
(faithfulness 2/5) across every iteration tested. Diagnosis: decomposition
fixed retrieval *coverage* (every quarter now gets searched), but auto-
generated sub-questions don't always retrieve as *precisely* as the
original, more specific phrasing did — the sub-question "What was NVIDIA's
total quarterly revenue in Q1 FY2027?" uses generic wording that doesn't
match how the transcript actually states the figure, while the original
compound question's implicit anchor to "Data Center revenue" retrieved
correctly every time it was asked as a single question. Coverage and
retrieval precision are two separable problems; this fix solved one, not
both. Deliberately left unfixed rather than tuned further: a fourth
iteration targeted at this one gold-set question risks overfitting the
decomposition prompt to this specific test case rather than generalizing —
the same failure mode as tuning a model against a test set it can see. A
held-out validation set (distinct from the 32-question gold set used to
develop this pipeline) would be the correct way to keep iterating safely
here, and is a natural v2 addition.

This table is the single most important artifact in the whole project — it's
the evidence that you improved something and can prove it, which is exactly
what separates this from a tutorial clone.

## Live system

The full pipeline runs end-to-end: FastAPI backend (retrieval + query
decomposition + generation) + Streamlit frontend (query interface +
observability dashboard). A real test query — "How did Copilot seats grow
from Q2 to Q4 FY2026?" — answered correctly in 7.4s at $0.00037, and
visibly demonstrated a live instance of the documented decomposition
limitation: the model split the question into sub-queries for the two
*endpoints* mentioned (Q2, Q4) rather than every quarter in the range,
producing a correct-but-incomplete answer (15M → 30M, skipping the Q3
data point). This is a genuine, observed example of the "coverage vs.
completeness" gap already documented above, not a new bug — decomposition
depends on how the sub-query LLM call interprets range language like
"from X to Y," and that interpretation isn't always exhaustive.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your OpenAI API key (required for generation, query decomposition, and
the LLM-judge eval — everything else runs locally/free):

```bash
export OPENAI_API_KEY="sk-..."     # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

**Run the consolidated app (recommended — this is what's deployed):**
```bash
streamlit run frontend/app.py
```

**Or run the two-service version (demonstrates a proper FastAPI service layer):**
```bash
uvicorn api.main:app --reload        # terminal 1
streamlit run frontend/dashboard.py  # terminal 2
```

**Re-run the eval harness** (retrieval + generation + LLM-judge across the
32-question gold set):
```bash
python eval/run_eval.py --gold_qa eval/gold_qa.json --chunks_dir data/processed/chunks/ --pattern "*_turns_overlap.json"
```

See `docs/` for corpus sourcing notes and the eval methodology writeup.