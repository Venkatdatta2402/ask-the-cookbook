# Judge audit: the 2026-10-30 run

How far can the judge's numbers be trusted? Every claim it extracted from the 23 RAG answers was checked by hand,
and the judge's marks were compared with that check afterwards.

**What this is, and isn't.** It isn't Critique Shadowing, where a domain expert marks whole answers pass or fail with
a critique and the judge is tuned until it agrees. It's an audit at the level of single claims, and it was done by
Claude (Opus 5.5) on the owner's behalf, not by the owner. The judge is gpt-oss-120b, the model that wrote the
answers.

## How the check was done

- **What:** all 182 claims in `answers_2026-10-30.jsonl`, as the judge's first call extracted them.
- **Against what:** each claim was read next to the 5 passages its answer was given (the text exactly as the model
  saw it) and the reference points' evidence lines.
- **Blind:** the judge's marks were hidden until all 182 were done.
- **Marks:** right; imprecise (correct in substance, worded loosely); addition (a part no passage states); wrong (it
  contradicts the cookbook).
- **Record:** every claim's mark is in [`../data/claim_checks_2026-10-30.json`](../data/claim_checks_2026-10-30.json).

## The answers

| Mark | Claims | Share |
|---|---|---|
| Right | 172 | 94.5% |
| Imprecise | 5 | 2.7% |
| Addition | 4 | 2.2% |
| **Wrong** | **1** | 0.5% |

**Wrong.** Q4 says "storing a memory costs essentially nothing". The cookbook says the opposite: *not* writing
something down costs nothing, which is why the model is left to decide what to keep.

**Additions,** parts no passage states:
- Q2 defines a doom loop as "any pattern" of repeated work, where the passages give four specific checks.
- Q16 adds a reason: "without the extra complexity of async code".
- Q19 gives "0.8 to 0.9" for a range the cookbook measures at 0.816.
- Q20 says a vector index "adds unnecessary complexity and latency".

**Imprecise,** correct in substance:
- Q1 says the model enforces the schema; it's the API.
- Q7 says the SDK warns about `env={**os.environ}`; it's the README.
- Q14 says the solver reads only the evidence; it reads the plan as well.
- Q27 says the Connection box is replaced; it's the session inside it.
- Q29 says tokens are tallied before each call; it's the budget that's checked before.

**Faithfulness by hand** (every part of a claim in the passages): 177 of 182 = **0.97**. The judge reported 0.98.

## The judge, against the check

Using the judge's grounding rule v3 and check v1, which produced the scores in `generation_2026-10-30.md`:

| | Judge |
|---|---|
| Of the 8 claims that misstate their passages (the wrong one, the 4 additions, and Q1, Q7, Q14), caught | **1** (Q2) |
| False alarms: claims flagged that are right | **5** of its 6 flags |
| The wrong claim marked contradicted | **no**: marked supported |

**What it missed:** a reversal in nearly the cookbook's own words; the right fact credited to the wrong thing; a
changed number; a widened scope; an added reason or cost.

**What it flagged wrongly:**
- claims backed by two cited passages together (Q3, Q21)
- a summary of several passages (Q27)
- a consequence a passage states outright (Q28: a newer release "breaks 07_mcp", so the pin keeps it working)
- a short label for what a passage describes (Q14)

**Noisy:** on borderline claims, the same claim came out differently between runs with nearly the same prompt.

**What that means for the scores:**
- The averages hold: faithfulness 0.98 reported against 0.97 by hand.
- "Claims wrong: 0" is an undercount; it's 1.
- A single flag is weak evidence: most are false, and most real problems aren't flagged.

## What was tried because of it, and didn't work

Telling the judge which kinds of claim count (grounding v2, v3) moved little. So a third attempt made it **show its
work** in calls 2 and 3 (grounding v4, check v2):
1. Split the claim into its parts: who, what, to what, numbers, scope words, reasons.
2. Copy the key's or the passages' own words for each part.
3. Compare part by part.

The prompts' examples were made up, not taken from these answers. Measured against the hand check:

| | Old prompts | Show your work |
|---|---|---|
| **Pilot**: the 15 claims behind the misses and false alarms | caught 1 of 8, 5 false alarms | caught 5 of 8, 3 false alarms |
| **Re-run**: 9 whole answers, 70 claims | caught 1 of 4, 1 false alarm | caught 1 of 4, **8** false alarms |

The pilot's gains didn't come back on whole answers. Claims caught in the pilot (Q4's reversal, Q7) were missed
again, so the gains were largely the judge's run-to-run noise. False alarms rose because quoting every part makes
the reply much longer. Under Groq's free tier, a request (prompt, reply room and format together) can't exceed
8,000 tokens a minute, and a passage-heavy prompt leaves the reply too little room. Q4 came back with 5 of its 11
claims unmarked even after a retry.

**Decision:** the judge was switched back to the old prompts (check v1, grounding v3). The reported numbers come from
them, and their averages held up against the hand check. For single claims, this audit is the trustworthy measure, not the judge's flags. Getting reliable
single flags would need more room per request (a paid tier) or a different judge.
