"""
LLM-as-judge evaluation harness.

Scores each (question, retrieved_chunks, generated_answer) triple on three
dimensions, each 1-5:

  faithfulness  — is every claim in the answer actually supported by the
                  retrieved chunks? (catches hallucination)
  relevance     — does the answer actually address what was asked?
  evasiveness_handling — for questions where management gave a hedged/vague
                  answer, does the system correctly represent that as
                  hedged, rather than inventing a confident answer that
                  wasn't said? (this is the earnings-call-specific check —
                  most generic RAG eval harnesses don't test for this)

Design note: the judge is given the gold answer AND the retrieved context,
so it can catch cases where the model answered fluently but ungrounded.
Keep the judge prompt strict and ask for structured JSON output so scores
are parseable and reproducible.
"""

import json
import os
from dataclasses import dataclass, asdict

from openai import OpenAI

client = OpenAI()  # requires OPENAI_API_KEY, or swap for Anthropic client

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of AI-generated answers about \
earnings call transcripts. You will be given a question, the source context \
the AI retrieved, the AI's generated answer, and (if available) a gold \
reference answer. Score the AI's answer on three dimensions from 1-5:

- faithfulness: Is every factual claim in the answer directly supported by \
the provided context? A claim not present in the context, even if plausible \
or "probably true", scores low. 5 = fully grounded, 1 = fabricated.
- relevance: Does the answer address what was actually asked, at the right \
level of specificity? 5 = directly on point, 1 = off-topic or non-responsive.
- evasiveness_handling: If the source context shows management gave a vague, \
hedged, or non-committal answer, did the AI correctly represent that \
uncertainty rather than inventing a confident specific answer? If the \
context contains a clear direct answer, this dimension should score 5 by \
default (not applicable as a penalty). 1 = AI invented false confidence \
where the source was actually evasive.

Respond with ONLY valid JSON in this exact shape, no other text:
{"faithfulness": <int>, "relevance": <int>, "evasiveness_handling": <int>, \
"reasoning": "<one or two sentence justification>"}
"""


@dataclass
class JudgeScore:
    question_id: str
    faithfulness: int
    relevance: int
    evasiveness_handling: int
    reasoning: str


def judge_answer(
    question: str,
    retrieved_context: str,
    generated_answer: str,
    gold_answer: str | None = None,
    question_id: str = "unscored",
    model: str = "gpt-4o-mini",
) -> JudgeScore:
    user_content = f"""Question: {question}

Retrieved context:
{retrieved_context}

AI's generated answer:
{generated_answer}

Gold reference answer (if available): {gold_answer or "N/A"}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    return JudgeScore(
        question_id=question_id,
        faithfulness=parsed["faithfulness"],
        relevance=parsed["relevance"],
        evasiveness_handling=parsed["evasiveness_handling"],
        reasoning=parsed["reasoning"],
    )


def run_eval_suite(results: list[dict]) -> dict:
    """
    results: list of dicts, each with keys:
        question_id, question, retrieved_context, generated_answer, gold_answer (optional)

    Returns aggregate scores + per-question detail, ready to log for the
    results table in the README and the Streamlit dashboard.
    """
    scores = []
    for r in results:
        score = judge_answer(
            question=r["question"],
            retrieved_context=r["retrieved_context"],
            generated_answer=r["generated_answer"],
            gold_answer=r.get("gold_answer"),
            question_id=r["question_id"],
        )
        scores.append(score)

    n = len(scores)
    aggregate = {
        "n_questions": n,
        "avg_faithfulness": sum(s.faithfulness for s in scores) / n if n else 0,
        "avg_relevance": sum(s.relevance for s in scores) / n if n else 0,
        "avg_evasiveness_handling": sum(s.evasiveness_handling for s in scores) / n if n else 0,
    }
    return {"aggregate": aggregate, "per_question": [asdict(s) for s in scores]}


if __name__ == "__main__":
    # Smoke test with a fabricated example — replace with real pipeline output
    example = [{
        "question_id": "smoke_test_1",
        "question": "Did the CFO commit to a specific margin target for next quarter?",
        "retrieved_context": "CFO: 'We're seeing encouraging trends but it's too early to give specific guidance on margins for next quarter.'",
        "generated_answer": "The CFO committed to a 3% margin improvement next quarter.",
        "gold_answer": "The CFO did not commit to a specific target; the answer was hedged.",
    }]
    print(json.dumps(run_eval_suite(example), indent=2))
