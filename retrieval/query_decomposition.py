"""
Query decomposition for cross-quarter questions.

The Step 6 eval surfaced a specific, diagnosable bug: cross-quarter
questions (e.g. "how did Copilot seats grow from Q2 to Q4") were retrieved
with a SINGLE query vector, and that one query structurally couldn't
represent "I need data from four different quarters" -- in practice this
meant one quarter's chunk (often the most recent, most distinctively
worded one) dominated the top-k, and other required quarters' chunks never
made it into context at all. The generated answer was then, correctly,
grounded only in what it received -- it just never received everything it
needed.

The fix: detect when a question spans multiple time periods, split it into
one sub-question per period, and retrieve SEPARATELY for each -- guaranteeing
every period gets its own retrieval slots rather than competing in one
shared ranked list where a single period can crowd out the others.

For a normal single-period question, decomposition is a no-op: it returns
the original question as the only "sub-question", so retrieval behavior for
direct_factual and evasiveness_test questions is completely unchanged.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

client = OpenAI()

DECOMPOSITION_SYSTEM_PROMPT = """You analyze questions about company earnings \
calls to determine if they require information from MULTIPLE different \
fiscal quarters/time periods to answer completely, or just ONE quarter (or \
no specific quarter at all).

CRITICAL DISTINCTION: only split a question if it spans multiple DIFFERENT \
TIME PERIODS. A question that asks about several different facts, metrics, \
or figures — but all within the SAME single quarter — is NOT a
multi-period question and must NOT be split. Splitting a single-quarter
question into "fact A" and "fact B" sub-questions loses the shared context
that anchors both facts to that specific quarter, and can cause retrieval
to drift into a different quarter entirely for one of the sub-questions.
When in doubt, do NOT split — a false negative (not splitting when it might
have helped slightly) is far less costly than a false positive (splitting
a single-quarter question and corrupting its retrieval).

If the question needs only one quarter (or is general/doesn't reference a \
specific time period), respond with a JSON list containing ONLY the \
original question unchanged -- even if that question asks about several \
different metrics or facts within that one quarter.

If the question spans multiple quarters (e.g. asks how something changed, \
grew, evolved, or compares across quarters), break it into separate, \
self-contained sub-questions -- one per quarter mentioned or implied. Each \
sub-question must be answerable independently and should mention the \
specific quarter and company explicitly, since it will be used standalone \
as a search query.

Respond with ONLY a valid JSON array of strings, no other text.

Example input: "How did Microsoft's Copilot seats grow from Q2 to Q4 FY2026?"
Example output: ["What were Microsoft's Copilot paid seats in Q2 FY2026?", \
"What were Microsoft's Copilot paid seats in Q3 FY2026?", \
"What were Microsoft's Copilot paid seats in Q4 FY2026?"]

Example input: "What was Azure revenue growth in Q4 FY2026?"
Example output: ["What was Azure revenue growth in Q4 FY2026?"]

Example input: "What was NVIDIA's Data Center revenue for Q1 FY2027, and \
what was its year-over-year growth rate?"
Example output: ["What was NVIDIA's Data Center revenue for Q1 FY2027, and \
what was its year-over-year growth rate?"]
(NOT split -- both facts belong to the SAME quarter; splitting would risk
losing that shared quarter context during retrieval.)

Example input: "What did JPMorgan report for net income and ROTCE for Q1 2026?"
Example output: ["What did JPMorgan report for net income and ROTCE for Q1 2026?"]
(NOT split -- two metrics, but ONE quarter.)
"""


import re

QUARTER_PATTERN = re.compile(r"Q[1-4]\s*(?:FY)?\s*20\d{2}|20\d{2}\s*Q[1-4]", re.IGNORECASE)


def _distinct_quarters_mentioned(sub_questions: list[str]) -> int:
    quarters = set()
    for q in sub_questions:
        for match in QUARTER_PATTERN.findall(q):
            quarters.add(re.sub(r"\s+", "", match.upper()))
    return len(quarters)


def decompose_query(question: str, model: str = "gpt-4o-mini") -> list[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()
    # Strip markdown code fences if the model wrapped the JSON in them
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        sub_questions = json.loads(content)
        if not (isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions)):
            return [question]
    except (json.JSONDecodeError, TypeError):
        return [question]

    if len(sub_questions) <= 1:
        return sub_questions

    # Structural safeguard, independent of the prompt: a valid multi-period
    # split must reference at least 2 DISTINCT quarters across its
    # sub-questions. If the model split a single-quarter, multi-fact
    # question (e.g. "revenue AND growth rate" for the same quarter) despite
    # being told not to, the sub-questions will reference the same quarter
    # (or none at all) -- catch that here and fall back to the original
    # question rather than trust the prompt alone. This is what actually
    # prevents the regression seen when the model over-split single-quarter
    # questions: relying on prompt instructions alone wasn't reliable enough
    # on its own, verified by observing exactly that failure in testing.
    if _distinct_quarters_mentioned(sub_questions) < 2:
        return [question]

    return sub_questions


def decomposed_retrieve(retriever, question: str, top_k_per_subquery: int = 3, decompose_model: str = "gpt-4o-mini"):
    """
    Retrieves chunks for a question, decomposing into per-quarter sub-queries
    first if needed. Returns a deduplicated list of RetrievedChunk objects
    that GUARANTEES each sub-query contributes its own top_k_per_subquery
    results, rather than pooling everything into one shared ranked list
    where a single dominant sub-query could crowd out the others -- that
    crowding-out is exactly what caused the original bug.
    """
    sub_questions = decompose_query(question, model=decompose_model)

    seen_ids = set()
    merged = []
    for sub_q in sub_questions:
        results = retriever.search(sub_q, top_k=top_k_per_subquery, rerank=False)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                merged.append(r)

    return merged, sub_questions