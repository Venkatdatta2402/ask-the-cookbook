"""SPENDS GROQ TOKENS when run. A judge for the saved RAG answers, claim by claim, giving five numbers an answer.
Ported from the cookbook's 08_frameworks/07_langsmith/claim_judge.py, whose trials found why each step is there.

    1. extract  the answer's claims, each with its own words quoted and the passages it cites -- shown the question
                and the answer only
    2. check    against the key: each claim supported, contradicted or not in the key, by the reference points and
                their evidence lines; whether it is on the question; and which reference points the answer makes
    3. ground   against what the answer was written from: whether a passage it cites says it, another given passage
                does, or none does -- shown the passages and the claims, not the key

    claims wrong        claims the key contradicts                         (2)
    faithfulness        claims some given passage backs, of all claims      (3)
    relevance           claims on the question, of all claims               (2)
    completeness        reference points the answer makes, of all points    (2)
    citation accuracy   cited claims whose cited passage backs them, of cited claims   (1, 3)

Why claim by claim (07_langsmith). Must-contain words and cited files passed an answer that put facts on the wrong
side of a comparison, and a verdict on the whole answer ("is this faithful?") has the same blind spot: most of it is
right. A claim that attaches the right fact to the wrong thing is one claim, and it is wrong.

Why extracting is its own call (07_langsmith). Asked to list an answer's claims while looking at the answer key, the
judge listed the KEY: it rewrote the claims into the reference points' shape and left out every claim the key had no
point for, among them one that was wrong. So extracting never sees the key, and every claim quotes the answer -- a
quote that is not in the answer is caught by code (`unanchored`), as are claims left unmarked.

Why the key and the passages are separate calls. They measure different things -- what is TRUE of the cookbook, and
what the answer was GIVEN -- and a true claim from a passage outside the key is "not in the key" yet backed. Together
they do not fit one request: Groq's free tier refuses one over 8,000 tokens.

What the numbers rest on. Every one of them is the judge's marking, and the judge is gpt-oss-120b, the model that
wrote the answers (the owner's choice, 2026-10-30). Where a claim ends is the judge's call too, so the shares move
with how finely it splits; completeness does not, being read off the whole answer. So the owner checks a sample of
the marks by hand (review.py), and the report gives how many were right beside the numbers.

Sizes. Each prompt asks for what it leaves of the 8,000, and the claims are checked in batches whose prompts stay
within `REQUEST_TOKENS`, each batch with the whole of what it is checked against. Measured offline on the 23 answered
RAG questions, with their real passages, a 700-token answer and 15 claims: extract ~920 tokens in, check ~2,270,
ground ~2,820, one call each for every question -- so with the replies (reasoning included) roughly 9,000-10,000 an
answer, ~220,000 for the 23. The daily 200,000 is shared with answering: the file is saved after every answer, and a
run stopped by the daily limit carries on where it stopped.

    python -m evals.generation.claim_judge <answers.jsonl>              every answer not yet judged, to verdicts_<date>.json
    python -m evals.generation.claim_judge <answers.jsonl> --rejudge    also calls 2 and 3 again where their rules changed,
                                                                        the extracted claims kept
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parents[1]
GOLDEN = HERE / "data" / "generation_golden.json"

GROQ_LIMIT = 8000          # tokens in one request on Groq's free tier: a larger one is refused (413)
REQUEST_TOKENS = 4500      # what one prompt is kept to, so the reply (its reasoning counts) has room


# --------------------------------------------------------------------------- 1. extract

class Claim(BaseModel):
    claim: str = Field(description="One statement the answer makes about the repo, in a few plain words")
    quote: str = Field(description="The answer's own words that make it, copied exactly: a phrase or sentence")
    cites: list[int] = Field(description="The passage numbers the answer attaches to these words -- a marker inside "
                                         "them or closing their sentence -- or [] if none")


class Claims(BaseModel):
    claims: list[Claim]


EXTRACT = """Below is an answer to a question about a code repository. List every claim it makes about the
repository: what something does, which part does it, a number, a measured result, a reason, a limitation, a file or
line it points to -- including claims inside tables, lists and code comments, and including claims about which file
a piece of code comes from. One claim per statement; a table cell or bullet that says two things is two claims.
Skip headings, restatements of the question, and general advice that says nothing about this repository.

For each claim, copy the answer's own words that make it, exactly as they appear. The answer cites the passages it
was written from with markers such as [1], [1, 2], [Source 2] or 【3†L1-L4】 (that one cites passage 3): a marker is
not a claim, but give each claim the passage numbers its marker cites.

QUESTION
{question}

ANSWER
{answer}"""

# Live, on the first day: a 1,130-character answer making eight plain claims (Q3, refused against broken tools) came
# back as {"claims": []} -- and with no claims, nothing was checked, so it scored completeness 0. An answer with text
# makes claims, so an empty list is a slip: asked again, with this said, and left unjudged if it persists.
EMPTY = """

A first reading of this answer listed no claims at all, but the answer makes claims about the repository. List every
one of them."""
EXTRACT_TRIES = 3


class NoClaims(RuntimeError):
    """The extract call listed no claims for an answer that has text, every time it was asked."""


# --------------------------------------------------------------------------- 2. check, against the key

class Mark(BaseModel):
    n: int = Field(description="The claim's number")
    verdict: Literal["supported", "contradicted", "not in the key"]
    source: str = Field(description="What it was checked against: evidence lines (e.g. 'runtime.py 185') or a "
                                    "reference point (e.g. 'point 3'); empty when not in the key")
    why: str = Field(description="For a contradicted claim, what the key says instead; otherwise empty")
    on_question: bool = Field(description="True if the claim helps answer the question asked")


class Marks(BaseModel):
    marks: list[Mark]
    points_made: list[int] = Field(description="The numbers of the reference points the answer makes")


CHECK = """You check claims made in an answer about a code repository against the repository's own words.

QUESTION
{question}

REFERENCE POINTS -- what a right answer says. Each is true of the repo.
{points}

EVIDENCE -- the repo's lines the reference points come from, numbered.
{evidence}

THE ANSWER
{answer}

ITS CLAIMS, numbered, each with the answer's words that make it
{claims}

1. Mark every numbered claim, once:
   supported       the evidence or a reference point says it. Plausible is not enough: what neither of them
                   actually says is "not in the key"
   contradicted    the evidence or a reference point says otherwise. This includes the right fact attached to the
                   wrong thing -- something one design does, credited to the other; a feature one has, said to be
                   missing from it; code from one file, said to be from another -- a wrong number, and the reverse
                   of what is said
   not in the key  neither says it, nor says otherwise
   Read each claim as the answer makes it, in its place: a claim under a heading or in a column about one design
   is about that design, and in an answer comparing two things, "only X" means only X of those two.
   Citation markers ([1], [Source 2], 【3†L1-L4】) number the passages the answer was given. They are not file
   names or line numbers, and a claim is not wrong for carrying one.
2. For every claim, say whether it is on the question: true if it helps answer what was asked, false if it is about
   something the question did not ask, however true.
3. List the reference points the answer makes. A point counts when the answer states its substance, in any words;
   not when it only mentions the topic.

Judge only against the evidence and the reference points, not against what you know of other software."""

# Version 1. A version 2 that compared part by part and copied the key's words was tried on 2026-10-31, after a hand
# check of all 182 claims of the first run, and reverted: it caught no more on whole answers and raised more false
# alarms (reports/judge_audit_2026-10-30.md).
CHECK_VERSION = 1


# --------------------------------------------------------------------------- 3. ground, against the passages

class Ground(BaseModel):
    n: int = Field(description="The claim's number")
    backed: Literal["cited", "other", "none"]
    passage: int = Field(description="For 'other', the number of the passage that says it; otherwise 0")
    why: str = Field(description="For 'none', what the passages say instead, or that they are silent; otherwise empty")


class Grounds(BaseModel):
    marks: list[Ground]


GROUND = """You check whether the claims an answer makes are backed by the passages it was written from.

THE PASSAGES, numbered as the answer cites them
{passages}

THE CLAIMS, numbered, each with the answer's words that make it and the passages it cites
{claims}

Mark every numbered claim, once. Read the passages it cites first:
   cited   a passage it cites says it -- even if another passage says it too
   other   no passage it cites says it, but another passage above does -- also for a claim that cites nothing
   none    no passage above says it, whether or not it is true
A passage says a claim when it states the claim's substance, in any words: a paraphrase, a summary or a short label
for what it says, and a conclusion the passage itself draws all count. But check every part of the claim. A claim
that changes what the passage says -- reverses it, or attaches it to something else -- is not backed by it. And a
claim that adds a detail no passage has (a cause, a cost, a number, a property) is "none" even when the rest of it is
backed: say in `why` which detail. Plausible is not enough, and neither is what is true of software in general: only
what these passages say counts."""

# Version 3 (2026-10-30), after the first run's review: of 11 claims the judge flagged, 8 were backed by the very
# passage they cited -- paraphrases, summaries, and a conclusion the passage drew, marked other or none -- so the rule
# says what counts (v2). Tried on six answers, v2 fixed five of those and let through a claim adding a detail no passage
# had (a cost); v3 makes every part of a claim count, which is what faithfulness is for.
# A version 4 that made the judge take each claim apart and quote the passages for every part was tried on
# 2026-10-31 and reverted: on whole answers it caught no more and raised more false alarms, its longer replies not
# fitting Groq's 8,000 a minute (reports/judge_audit_2026-10-30.md).
GROUND_VERSION = 3


def numbered_points(question):
    return "\n".join(f"{i}. {p['point']}" for i, p in enumerate(question["points"], 1))


def evidence(question):
    """The question's evidence lines, numbered, under the file they come from -- the indexed text, so a notebook is
    numbered by its prose."""
    from ask_the_cookbook import search
    parts = []
    for file, first, last in question["evidence"]:
        lines = search.file_text(file).splitlines()
        body = "\n".join(f"{n:>4}  {lines[n - 1]}" for n in range(first, min(last, len(lines)) + 1))
        parts.append(f"[{file}, lines {first}-{last}]\n{body}")
    return "\n\n".join(parts)


def passages(sources):
    """The passages the answer was given, as it was given them (collect_answers keeps the text)."""
    missing = [s["n"] for s in sources if not s.get("text")]
    if missing:
        raise ValueError(f"sources {missing} have no text: collect the answers again (collect_answers keeps it)")
    return "\n\n".join(f"--- passage [{s['n']}]: {s['file']}, lines {s['lines'][0]}-{s['lines'][1]}\n"
                       f"{s['text'].strip()}" for s in sources)


def extract_prompt(question, answer):
    """Extracting is shown the question and the answer, and nothing of the key or the passages."""
    return EXTRACT.format(question=question["question"], answer=answer)


def check_prompt(question, answer, claims):
    listed = "\n".join(f"{i}. {c['claim']}\n   words: \"{c['quote']}\"" for i, c in enumerate(claims, 1))
    return CHECK.format(question=question["question"], points=numbered_points(question),
                        evidence=evidence(question), answer=answer, claims=listed)


def ground_prompt(sources, claims):
    """Grounding is shown the passages and the claims: not the key, which would make it a second check."""
    listed = "\n".join(f"{i}. {c['claim']}\n   words: \"{c['quote']}\"\n   cites: "
                       f"{', '.join(map(str, c['cites'])) or 'nothing'}" for i, c in enumerate(claims, 1))
    return GROUND.format(passages=passages(sources), claims=listed)


# --------------------------------------------------------------------------- calling the model

_ENCODING = None


def size(text):
    """Tokens, as gpt-oss counts them (o200k_base), from the cache committed with the app."""
    global _ENCODING
    if _ENCODING is None:
        from ask_the_cookbook.runtime import chat  # noqa: F401 -- points tiktoken at the committed cache
        import tiktoken
        _ENCODING = tiktoken.get_encoding("o200k_base")
    return len(_ENCODING.encode(text))


def batches(prompt, claims, limit=None):
    """The claims in groups whose prompts -- `prompt(group)` -- fit one request, in order."""
    limit = limit or REQUEST_TOKENS
    groups, current = [], []
    for claim in claims:
        if current and size(prompt(current + [claim])) > limit:
            groups.append(current)
            current = []
        current.append(claim)
    return groups + ([current] if current else [])


class JudgeQuotaSpent(RuntimeError):
    """Groq's daily token limit for this key is spent. Nothing more is judged until it resets."""


def client(key):
    """A Groq client on `key`. Its per-minute 429s are waited out by the client itself (`max_retries`, honouring
    Retry-After): a judge's call is a few thousand tokens against a limit of 8,000 a minute."""
    from ask_the_cookbook import answers
    return answers.client(key)


def ask(groq, prompt, schema, model, log=print):
    """One structured call: the schema enforced by the API (`json_schema`), the reply validated here, and asked again
    if it does not validate. Returns (parsed, tokens spent on it, every try counted)."""
    import openai
    response_format = {"type": "json_schema",
                       "json_schema": {"name": schema.__name__.lower(), "schema": schema.model_json_schema()}}
    spent = 0
    for attempt in (1, 2, 3):
        try:
            # the reply gets what the request leaves of the 8,000. Left unset (tried 2026-10-31), Groq's own default was
            # too small for a reply whose reasoning counts as output: grounding replies came back cut off, then empty.
            # Groq counts the reply's room, the prompt AND the schema against its 8,000 a minute (live: "Requested
            # 8157" with 200 left over for the schema), so the schema is counted here too
            room = GROQ_LIMIT - size(prompt) - size(json.dumps(response_format)) - 300
            response = groq.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                response_format=response_format, max_tokens=room)
        except openai.RateLimitError as error:
            if "per day" in str(error):
                raise JudgeQuotaSpent(f"{model}: daily token limit spent") from error
            raise
        except openai.BadRequestError as error:
            # the API's own refusal of a reply that does not fit the schema (json_validate_failed) is the model's
            # slip, and asked again; any other 400 is a real error
            if attempt == 3 or "json_validate_failed" not in str(error):
                raise
            log(f"    reply did not fit the {schema.__name__} schema, asking again")
            continue
        spent += response.usage.total_tokens if response.usage else 0
        try:
            return schema.model_validate_json(response.choices[0].message.content), spent
        except ValueError:
            if attempt == 3:
                raise
            log(f"    reply was not a valid {schema.__name__}, asking again")


def marked(groq, prompt, schema, claims, model, log):
    """Every claim's mark from `schema`'s calls, batch by batch, keyed by the claim's number in the whole answer;
    the first batch's reply (for what it says of the answer as a whole); and the tokens spent."""
    marks, first, spent, start = {}, None, 0, 0
    for group in batches(prompt, claims):
        reply, used = ask(groq, prompt(group), schema, model, log)
        spent += used
        got = {m.n: m for m in reply.marks if 1 <= m.n <= len(group)}
        if len(got) < len(group):
            # live (2026-10-31): a grounding reply that fitted the schema yet marked none of Q4's 11 claims -- read
            # as "backed by nothing", it made faithfulness 0.18. Asked once more; the better reply is kept
            log(f"    {schema.__name__}: {len(group) - len(got)} of {len(group)} claims unmarked, asking again")
            again, used = ask(groq, prompt(group), schema, model, log)
            spent += used
            more = {m.n: m for m in again.marks if 1 <= m.n <= len(group)}
            if len(more) > len(got):
                reply, got = again, more
        marks.update({start + n: m for n, m in got.items()})
        first = reply if first is None else first
        start += len(group)
    return marks, first, spent


# --------------------------------------------------------------------------- one answer

_MARKER = re.compile(r"【[^】]*】|\[(?:Source\s*)?\d+(?:\s*,\s*(?:Source\s*)?\d+)*[^\[\]]*\]")


def _flat(text):
    # citation markers out, so a quote that skips one still counts as the answer's words
    return " ".join(re.sub(r"[*`_]", "", _MARKER.sub(" ", text or "")).lower().split())


def unanchored(claims, answer):
    """Claims whose quoted words are not in the answer (ignoring case, spacing, markdown marks and citations)."""
    return [c for c in claims if _flat(c["quote"]) not in _flat(answer)]


def claims(question, record, groq, model, log=print):
    """The answer's claims, each marked against the key and against its passages, and the reference points it
    makes: (found, tokens)."""
    answer, sources = record["answer"], record["sources"]
    found, spent = ask(groq, extract_prompt(question, answer), Claims, model, log)
    for _ in range(EXTRACT_TRIES - 1):
        if found.claims or not answer.strip():
            break
        log("    no claims listed for an answer with text, asking again")
        found, used = ask(groq, extract_prompt(question, answer) + EMPTY, Claims, model, log)
        spent += used
    if not found.claims:
        raise NoClaims(f"no claims listed in {EXTRACT_TRIES} tries ({spent:,} tokens)")
    listed = [c.model_dump() for c in found.claims]
    given = {s["n"] for s in sources}
    for c in listed:
        c["cites"] = sorted({n for n in c["cites"] if n in given})       # a number with no passage cites nothing
    found = {"claims": listed, "unanchored": len(unanchored(listed, answer))}
    spent += check(question, record, found, groq, model, log)
    return found, spent + ground(record, found, groq, model, log)


def _unmarked(found):
    found["unmarked"] = sum(c.get("verdict") == "unmarked" or c.get("grounded") == "unmarked" for c in found["claims"])


def check(question, record, found, groq, model, log=print):
    """Call 2 on claims already extracted: each one's marks against the key set (or replaced), the reference points
    the answer makes, and the tokens it cost. Its own function, as `ground` is, so that a change to the check re-runs
    this call alone."""
    claims, answer = found["claims"], record["answer"]
    marks, first, spent = marked(groq, lambda group: check_prompt(question, answer, group), Marks, claims, model, log)
    for i, c in enumerate(claims, 1):
        k = marks.get(i)
        c.update(verdict=k.verdict if k else "unmarked", source=k.source if k else "", why=k.why if k else "",
                 on_question=k.on_question if k else None)
    found["points made"] = sorted({p for p in (first.points_made if first else [])
                                   if 1 <= p <= len(question["points"])})
    found["check version"] = CHECK_VERSION
    _unmarked(found)
    return spent


def ground(record, found, groq, model, log=print):
    """Call 3 on claims already extracted and checked: each one's grounding marks set (or replaced), and the tokens it
    cost. Its own function so that a change to the grounding rule re-runs this call alone."""
    claims = found["claims"]
    marks, _, spent = marked(groq, lambda group: ground_prompt(record["sources"], group), Grounds, claims, model, log)
    for i, c in enumerate(claims, 1):
        g = marks.get(i)
        c.update(grounded=g.backed if g else "unmarked", passage=g.passage if g else 0, ground_why=g.why if g else "")
    found["ground version"] = GROUND_VERSION
    _unmarked(found)
    return spent


def scores(found, question):
    """The five numbers for one answer. A share over no claims is None, as is citation accuracy for an answer that
    cites nothing."""
    listed = found["claims"]
    share = lambda part, whole: round(part / whole, 3) if whole else None
    cited = [c for c in listed if c["cites"]]
    return {"claims": len(listed),
            "claims wrong": sum(c["verdict"] == "contradicted" for c in listed),
            "faithfulness": share(sum(c["grounded"] in ("cited", "other") for c in listed), len(listed)),
            "relevance": share(sum(c["on_question"] is True for c in listed), len(listed)),
            "completeness": share(len(found["points made"]), len(question["points"])),
            "citation accuracy": share(sum(c["grounded"] == "cited" for c in cited), len(cited))}


# --------------------------------------------------------------------------- every answer

def golden_by_id(path=GOLDEN):
    return {q["id"]: q for q in json.loads(Path(path).read_text(encoding="utf-8"))["questions"]}


def load_answers(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verdicts_path(answers_path):
    """reports/answers_<date>.jsonl -> reports/verdicts_<date>.json"""
    path = Path(answers_path)
    return path.with_name(path.stem.replace("answers", "verdicts", 1) + ".json")


def stale(found):
    """The calls whose rule changed since this answer was judged: "check" (call 2), "ground" (call 3), or neither."""
    if "claims" not in found:
        return []
    # a call that left claims unmarked is redone too: an unmarked claim counts as unbacked, so it must not stand
    unmarked = {"check": any(c.get("verdict") == "unmarked" for c in found["claims"]),
                "ground": any(c.get("grounded") == "unmarked" for c in found["claims"])}
    return [name for name, version in (("check", CHECK_VERSION), ("ground", GROUND_VERSION))
            if found.get(f"{name} version", 1) != version or unmarked[name]]


def judge_all(answers_path, groq, model, golden=None, rejudge=False, log=print):
    """Every answer not judged yet, saved after each one -- and with `rejudge`, calls 2 and 3 again on every answer
    judged under an older rule for them, its extracted claims kept (so a hand check of those claims still applies).
    Returns (judged, finished): finished is False when the daily limit stopped the run, and the rest waits."""
    golden = golden or golden_by_id()
    out = verdicts_path(answers_path)
    judged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    def save():
        out.write_text(json.dumps(judged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for record in load_answers(answers_path):
        qid, done = record["id"], judged.get(record["id"])
        again = stale(done) if rejudge and done is not None else []
        if done is not None and not again:
            continue
        if record["refused"]:
            judged[qid] = {"refused": True}
            continue
        try:
            if again:
                spent = 0
                if "check" in again:
                    spent += check(golden[qid], record, done, groq, model, log)
                if "ground" in again:
                    spent += ground(record, done, groq, model, log)
                done["tokens"] += spent
                found = done
            else:
                found, spent = claims(golden[qid], record, groq, model, log)
                judged[qid] = {**found, "tokens": spent, "judge": model}
        except JudgeQuotaSpent as stop:
            save()
            log(f"stopped: {stop}. Run again after the reset; nothing judged so far is lost.")
            return judged, False
        except NoClaims as empty:
            # not saved, so the next run tries it again; a zero here would read as an answer that said nothing
            log(f"{qid}: {empty}; left unjudged")
            continue
        save()
        numbers = scores(found, golden[qid])
        log(f"{qid}: {numbers['claims']} claims, {numbers['claims wrong']} wrong, faithfulness "
            f"{numbers['faithfulness']}, completeness {numbers['completeness']}, {spent:,} tokens")
    save()
    return judged, True


def main(argv):
    import os

    from ask_the_cookbook import search
    if not argv:
        print(__doc__)
        return 2
    _, finished = judge_all(argv[0], client(os.environ["GROQ_API_KEY"]), search.MODEL, rejudge="--rejudge" in argv)
    print(f"saved to {verdicts_path(argv[0])}")
    return 0 if finished else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
