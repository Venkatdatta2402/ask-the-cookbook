# Evaluation

How well the assistant finds and uses the cookbook, measured against answers worked out by hand.

| What | Spends | Runs |
|---|---|---|
| Retrieval against the golden set, with a baseline gate | nothing | every push (pytest) |
| Deterministic checks on saved answers | nothing | every push (on test answers); on real answers by hand |
| Collecting real answers | about 90k Groq tokens (no Gemini: kept embeddings) | by hand only |
| The judge: five numbers per answer, claim by claim | about 9-10k Groq tokens an answer | by hand only |
| The owner's check on a sample of the judge's marks | nothing | by hand |
| The report: checks, numbers, means, the owner's check | nothing | by hand |

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
- **Recall@5**: the share of the *needed* passages (grade 2 or 3) found in the 5 places. It is capped: a question
  needing 16 passages can't pass 5/16, so each report also gives the **best possible** recall, what a perfect
  search would score (RAG: 0.471 of a possible 0.687). Grade 1 is left out on purpose: counting it would lower the
  cap further and mostly measure how many grade-1 passages were listed.
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

Answers are collected the way the app writes them (same engine, model and prompt). The search is given each
question's kept embedding, so it sees exactly the passages the retrieval eval scores, and no Gemini call is made.

## Reference points (`data/generation_golden.json`)

For every RAG question, what a right answer says: a few points, each with quotes from the cookbook and the evidence
lines the quotes come from. Q1–Q10 were ported from the cookbook's `07_langsmith` (Q8–Q10 are agentic). The other
17 were written for this demo from the passages graded 3 and 2 in the golden set. The tests check that every quote
is in its evidence lines. Q14 has a `note`: two notebooks disagree on whether ReWOO runs in parallel, and its point
follows the later one.

## The judge (`generation/claim_judge.py`)

Ported from the cookbook's `07_langsmith` claim judge, on gpt-oss-120b (the model that wrote the answers: the
owner's choice). It reads an answer claim by claim, in three calls:

1. **Extract**: the answer's claims, each with its own words quoted and the passages it cites. This call sees the
   question and the answer only. Shown the reference points as well, the cookbook's judge rewrote the claims to match
   them.
2. **Check against the key**: each claim marked supported, contradicted, or not in the key (the reference points and
   their evidence lines); whether it's on the question; and which reference points the answer makes.
3. **Ground against the passages**: whether a passage the claim cites backs it, another given passage does, or none
   does. This call sees the 5 passages as the model was shown them, and not the key.

Code catches a quote that isn't in the answer and a claim left unmarked. From the marks, five numbers per answer:

| Number | What it counts | Measures |
|---|---|---|
| claims wrong | claims the key contradicts | correctness |
| faithfulness | claims some given passage backs, out of all claims | groundedness: nothing added beyond the context |
| relevance | claims on the question, out of all claims | no drifting off what was asked |
| completeness | reference points the answer makes, out of all points | nothing needed left out |
| citation accuracy | cited claims whose cited passage backs them, out of cited claims | citations that support what they're attached to |

The key and the passages are separate calls because they measure different things. A true claim taken from a passage
outside the key is "not in the key" but still backed. Together they wouldn't fit one request either: Groq refuses one
over 8,000 tokens. Measured on the 23 answered questions, with their real passages and an answer the size of a real
one: extract about 920 tokens in, check about 2,270, ground about 2,820, each one call. With the replies, that's
roughly 9,000-10,000 tokens an answer, about 220k for all 23, so about two days of Groq's free tier. Each answer's
marks are saved as it's judged, and a run stopped by the daily limit carries on where it stopped.

Where a claim ends is the judge's call, so the shares move a little with how finely it splits. Completeness doesn't,
because it's read off the whole answer.

## The judge audit ([`reports/judge_audit_2026-10-30.md`](reports/judge_audit_2026-10-30.md))

All 182 claims of the first run were checked by hand, blind to the judge's marks
(`data/claim_checks_2026-10-30.json`). Of the claims, 172 are right, 9 are imprecise or add a detail, and 1 is wrong.
The judge caught 1 of the 8 that misstate their passages, and 5 of its 6 flags were false. Its averages held up; its
single flags didn't. The show-your-work prompts above were tried because of it and didn't help on whole answers: no
more misstatements caught, and more false alarms. So the reported numbers use the earlier prompts.

## The owner's check (`generation/review.py`, `data/owner_review.json`)

Every number is the judge's own marking, so a sample of the marks is checked by hand:

- **Flagged**: every claim marked contradicted, backed by no passage, or off the question. Each of these lowers a
  number, so a wrong one is a false alarm.
- **Passed**: 20 of the claims marked fine on all three, chosen at random with a fixed seed. A wrong one there is a
  miss.

The sheet shows each claim with the answer's words, the judge's marks and reasons, and links to the key's evidence
and the passages it cites. The owner records whether each claim's marks are right, and the report gives how many
were, beside the numbers.

The judge runs check v1 and grounding v3, the versions behind the reported numbers. A later version that made the
judge **show its work** (split the claim into parts, quote the passages for each, then compare) was tried after the
judge audit (below) and reverted. It didn't help on whole answers, because its longer replies don't fit Groq's
free-tier limit. Each answer records which rule version judged it, and the report warns if they're mixed.
`--rejudge` re-runs calls 2 and 3 where a rule changed or a reply left claims unmarked, keeping the extracted
claims.

```bash
python -m evals.generation.claim_judge evals/reports/answers_<date>.jsonl    # SPENDS ~9-10k Groq tokens an answer
python -m evals.generation.claim_judge evals/reports/answers_<date>.jsonl --rejudge    # calls 2 and 3 again, ~9k an answer
python -m evals.generation.review evals/reports/answers_<date>.jsonl         # the sheet, and owner_review.json
python -m evals.generation.report evals/reports/answers_<date>.jsonl         # free: the numbers and the check
```
