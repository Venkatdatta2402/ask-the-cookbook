"""The owner's check on the judge. Every number the judge gives is built from its own marks on claims, so a sample of
those marks is looked at by hand, and how many were right is reported beside the numbers.

What is sampled, from a verdicts file:

    flagged   every claim the judge marked contradicted, backed by no passage, off the question, or citing a passage
              that does not back it. Each one lowers a number, so each is worth a look: a wrong flag is a false
              alarm
    passed    a fixed random sample of the claims it marked fine on all three counts, to see what it lets through:
              a wrong one there is a miss

The owner reads each on the sheet -- the claim, the answer's words, the judge's marks and reasons, links to the key's
evidence and to the passages the claim cites -- and records in data/owner_review.json whether the judge's marks on it
are right. Two other questions on the sheet are the owner's too, and are not sampled: whether the claims were split
sensibly, and whether the reference points are right.

    python -m evals.generation.review <answers.jsonl>     the sheet (reports/review_sheet_<date>.md) and, if there is
                                                          none yet, data/owner_review.json to fill in
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REVIEW = HERE / "data" / "owner_review.json"
SAMPLE, SEED = 20, 30

HOW = ("For each claim: \"right\": true if every mark the judge gave it is right, false if any is wrong -- and then "
       "\"note\": which, and why.")


def flags(claim):
    """What the judge said against a claim: each of these lowers a number."""
    out = []
    if claim["verdict"] == "contradicted":
        out.append("contradicted by the key")
    if claim["grounded"] == "none":
        out.append("backed by no passage")
    if claim["on_question"] is False:
        out.append("off the question")
    if claim["cites"] and claim["grounded"] != "cited":
        # first missed on the live run: Q4's citation accuracy of 0.50 rested on marks nobody would have looked at
        out.append("its cited passage does not back it")
    return out


def sample(judged, size=SAMPLE, seed=SEED):
    """[(key, qid, claim number, claim, "flagged" | "passed")]: every flagged claim, and `size` of the others chosen
    at random -- the same ones every time for the same verdicts."""
    flagged, passed = [], []
    for qid, found in judged.items():
        for i, c in enumerate(found.get("claims", []), 1):
            (flagged if flags(c) else passed).append((f"{qid}#{i}", qid, i, c))
    chosen = random.Random(seed).sample(passed, min(size, len(passed)))
    order = lambda item: (int(item[1][1:]), item[2])
    return sorted([(*f, "flagged") for f in flagged] + [(*p, "passed") for p in chosen], key=order)


def template(items, verdicts_name):
    return {"how": HOW, "verdicts": verdicts_name,
            "items": {key: {"claim": c["claim"], "kind": kind, "right": None, "note": ""}
                      for key, _, _, c, kind in items}}


def load(path=REVIEW):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sheet(items, records, golden):
    """The sheet to check from, as markdown: per sampled claim, what the judge said and where to look."""
    from ask_the_cookbook import answers
    by_id = {r["id"]: r for r in records}
    out = ["# Checking the judge's marks", "",
           f"{len(items)} claims: every one the judge marked against an answer, and "
           f"{sum(kind == 'passed' for *_, kind in items)} it passed, chosen at random. {HOW} "
           "Write it into `evals/data/owner_review.json`.", "",
           "Also worth a word in a note: a claim split oddly, or a reference point that is wrong.", ""]
    for key, qid, i, c, kind in items:
        q, record = golden[qid], by_id[qid]
        sources = {s["n"]: s for s in record["sources"]}
        out += [f"## {key} ({kind})", "", f"*{q['question']}*", "", f"**Claim:** {c['claim']}", "",
                f"> {c['quote']}", "",
                f"- against the key: **{c['verdict']}**" + (f" ({c['source']})" if c["source"] else "") +
                (f" -- {c['why']}" if c["why"] else ""),
                f"- on the question: **{'yes' if c['on_question'] else 'no'}**",
                f"- against the passages: **{c['grounded']}**" +
                (f" (passage {c['passage']})" if c["grounded"] == "other" else "") +
                (f" -- {c['ground_why']}" if c["ground_why"] else ""),
                "- the key's evidence: " + ", ".join(
                    f"[{f} {a}-{b}]({answers.file_link(f, a, b)})" for f, a, b in q["evidence"])]
        cited = [sources[n] for n in c["cites"] if n in sources]
        if cited:
            out.append("- the passages it cites: " + ", ".join(
                f"[[{s['n']}] {s['file']} {s['lines'][0]}-{s['lines'][1]}]"
                f"({answers.file_link(s['file'], *s['lines'])})" for s in cited))
        out.append("")
    return "\n".join(out)


def accuracy(review):
    """How many of the judge's reviewed marks were right, flagged and passed apart: (right, reviewed) for each."""
    out = {"flagged": [0, 0], "passed": [0, 0]}
    for item in review["items"].values():
        if item["right"] is None:
            continue
        out[item["kind"]][0] += item["right"]
        out[item["kind"]][1] += 1
    return {kind: tuple(counts) for kind, counts in out.items()}


def main(argv):
    from evals.generation import claim_judge as J
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    judged = json.loads(J.verdicts_path(path).read_text(encoding="utf-8"))
    items = sample(judged)
    out = path.with_name(path.stem.replace("answers", "review_sheet", 1) + ".md")
    out.write_text(sheet(items, J.load_answers(path), J.golden_by_id()), encoding="utf-8")
    print(f"sheet: {out}")
    if REVIEW.exists():
        print(f"{REVIEW} exists, left as it is")
    else:
        REVIEW.write_text(json.dumps(template(items, J.verdicts_path(path).name), indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(f"to fill in: {REVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
