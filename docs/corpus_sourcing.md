# Sourcing the transcript corpus

You need raw earnings call transcripts for the 5 companies (MSFT, NVDA, JPM,
COST, LUV), ideally 4-8 quarters each so the gold eval set has enough range.

## Free sources (pick one or combine)

1. **Company Investor Relations pages** — most large-caps post transcripts or
   at least prepared-remarks PDFs directly. Fastest, cleanest text, zero
   scraping risk. Start here.
2. **Motley Fool Earnings Call Transcripts** (free, no login for most) —
   consistently formatted, includes the full Q&A section with analysts named,
   which is important for your speaker-tagging step.
3. **SEC EDGAR 8-K exhibits** — some companies file transcripts as exhibits.
   Bonus: same EDGAR pull could later feed the 10-K numbers if you extend
   this project.
4. **Seeking Alpha free tier** — good quality, occasionally paywalled after a
   few reads per month.

## What to save

Save each transcript as a plain `.txt` or `.pdf` file in `data/raw/` using this
naming convention so the ingestion script can parse company/quarter
automatically:

```
data/raw/MSFT_2025_Q4.txt
data/raw/MSFT_2026_Q1.txt
data/raw/NVDA_2025_Q4.txt
...
```

## Why manual sourcing (not an automated scraper) for v1

Scraping investor-relations or transcript sites reliably needs per-site
parsing logic and is fragile — not a good use of your limited build time for
v1. Manually downloading ~30-40 transcripts (5 companies x 6-8 quarters) is a
one-time ~2 hour task. If you want to extend the project later, an automated
ingestion job (e.g. a scheduled scraper) is a legitimate "v2" feature to add
to the README roadmap — mention it as a stated next step even before you build
it; interviewers respond well to a clear-eyed "here's what I'd do with more
time" answer.

## Gold eval set

As you read through transcripts to save them, jot down 6-10 candidate Q&A
pairs per company where you already know the answer from reading the text.
This becomes your `eval/gold_qa.json` set later — writing them *while reading*
is far faster than doing it as a separate pass.

Aim for a mix of:
- Direct factual questions ("What was NVIDIA's data center revenue in Q4 2025?")
- Questions where management gave a **hedged/non-answer** (good for testing
  evasiveness detection)
- Cross-quarter questions ("How did Costco's membership fee revenue change
  from Q1 to Q2?") — these require retrieving from *multiple* chunks/quarters,
  which is a meaningfully harder retrieval test than single-chunk lookup.
