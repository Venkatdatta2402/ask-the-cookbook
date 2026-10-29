# Evaluation

How well the assistant finds and uses the cookbook, measured against answers worked out by hand.

| What | Spends | Runs |
|---|---|---|
| Retrieval against the golden set, with a baseline gate | nothing | every push (pytest) |
| Deterministic checks on saved answers | nothing | every push (on test answers); on real answers by hand |
| Collecting real answers | Groq tokens, Gemini embeddings | by hand only |

## The golden set (`data/retrieval_golden.json`)

30 questions (24 RAG, 6 agentic). Every chunk in the index (1,063 chunks, 124 files, cookbook `43c7bf5`) was read once,
with no search involved, and graded against all 30. Searching for candidates first would have graded only what the
searches found, and missed the rest.

| Grade | Meaning |
|---|---|
| 3 | states the answer, or a core part of it, directly |
| 2 | a needed supporting part of the answer |
| 1 | related: says something about the topic that helps a little |
| 0 | not relevant (every chunk not listed), including text that only names the topic |

510 graded passages. Each one quotes the chunk's own text and says why it got its grade. The tests check every
passage is a real indexed chunk and every quote is in it.

`data/query_embeddings.json` keeps each question's Gemini embedding (`RETRIEVAL_QUERY`, 768 dims), so retrieval runs
offline and gives the same numbers every time.

## Retrieval metrics (k = 5)

The search measured is the one RAG answers from: `CookbookRetriever` (vector + BM25, fused, behind the floor).

- **Matching.** A retrieved passage takes the grade of a golden one in the same file whose lines overlap it by at
  least half (Jaccard ≥ 0.5). Each golden passage is credited once.
- **Precision@5**: the share of the 5 places holding a passage of grade 1 or more.
- **Recall@5**: the share of the *needed* passages (grade 2 or 3) found in the 5 places.
- **MRR**: 1 / the place of the first relevant passage.
- **NDCG@5**: every grade, as gain 2^grade − 1, against the best order the golden set allows.

```bash
python -m evals.retrieval.evaluate_retrieval              # report; compares with the baseline
python -m evals.retrieval.evaluate_retrieval --baseline   # accept the current numbers as the baseline
```

The baseline (`reports/retrieval_baseline.json`) is committed, and `tests/test_retrieval.py` fails if any mean falls
below it. Re-baseline only on purpose, after reading what changed.

When the refresh Action moves the index to a new cookbook commit, the exact-span checks and the gate skip with a
message: re-check the golden set against the new chunks, then re-baseline.

## Checks on answers (`generation/deterministic_checks.py`)

Plain code, no model: the answer is not empty; it cites; every `[n]` is a listed source; every source is real lines
of a real file; and at least one cited source is a grade-3 passage. A refusal of a covered question fails. Whether
the answer is *true* is a judge's question, not these.

```bash
python -m evals.generation.collect_answers     # SPENDS ~90k Groq tokens for the 24 RAG questions
python -m evals.run_evals evals/reports/answers_<date>.jsonl
```

`data/generation_golden.json` holds the 10 questions ported from the cookbook's `07_langsmith`, with the points a
good answer makes and the quotes behind them, for the judge that comes later.
