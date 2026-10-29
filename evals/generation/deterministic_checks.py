"""Checks on a RAG answer that need no model: plain code, the same result every time, free to run on every answer.

They check what can be checked without judging the prose -- that the answer is there, that it cites, that what it
cites exists and was what the search handed it, and that its sources land where the golden set says the answer
is. Whether the answer is TRUE is a judge's question (and a person's), not these.

An answer is a dict, as `collect_answers.py` saves one:

    {"id": "Q3", "question": ..., "refused": False,
     "answer": the model's text, citations as it wrote them ([1], [Source 2], 【3†L1-L4】),
     "sources": [{"n": 1, "file": "03_tools/invoke.py", "lines": [1, 86]}, ...]}

    python -m evals.generation.deterministic_checks <answers.jsonl>     each answer's checks, and the totals
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GOLDEN = HERE / "data" / "retrieval_golden.json"


def cited(answer):
    """The source numbers an answer cites, in order, outside code -- parsed as the app parses them."""
    from ask_the_cookbook import answers
    found = []
    answers.citations(answer, lambda numbers, rest: found.extend(numbers))     # returns None: the text is unchanged
    return found


def file_lines(file):
    """How many lines a cookbook file has, as the index counts them (a notebook by its prose), or None if it is not
    one of the indexed files."""
    from ask_the_cookbook import answers, search
    if file not in answers.FILES:
        return None
    return search.file_text(file).count("\n") + 1


def check(record, golden=None):
    """{check: (passed, why)} for one answer. A refused answer is checked only for whether it should have been."""
    from evals.metrics.ir_metrics import jaccard
    results = {}
    expected = (golden or {}).get(record["id"])
    if record.get("refused"):
        results["answered"] = (expected is None, "refused" + (" a question the golden set says is covered"
                                                             if expected is not None else ""))
        return results
    text = (record.get("answer") or "").strip()
    results["answered"] = (bool(text), "an answer" if text else "empty")

    numbers = cited(text)
    listed = {s["n"] for s in record["sources"]}
    results["cites"] = (bool(numbers), f"{len(numbers)} citation(s)" if numbers else "no citation")
    unknown = sorted(set(numbers) - listed)
    results["citations resolve"] = (not unknown, "every [n] is a listed source" if not unknown
                                    else f"cites {unknown}, not among sources {sorted(listed)}")

    bad = []
    for s in record["sources"]:
        size = file_lines(s["file"])
        if size is None:
            bad.append(f"{s['file']} is not an indexed file")
        elif not 1 <= s["lines"][0] <= s["lines"][1] <= size:
            bad.append(f"{s['file']} lines {s['lines']} outside 1-{size}")
    results["sources exist"] = (not bad, "; ".join(bad) or "every source is real lines of a real file")

    if expected is not None:
        answer_passages = [(r["file"], tuple(r["lines"])) for r in expected if r["grade"] == 3]
        used = [(s["file"], tuple(s["lines"])) for s in record["sources"] if s["n"] in set(numbers)]
        hit = [f"{f} {list(l)}" for f, l in used if any(f == gf and jaccard(l, gl) > 0 for gf, gl in answer_passages)]
        results["cites an answer passage"] = (bool(hit), ", ".join(hit) or
                                              f"none of the {len(answer_passages)} grade-3 passages is cited")
    return results


def golden_by_id(path=GOLDEN):
    golden = json.loads(Path(path).read_text(encoding="utf-8"))
    return {q["id"]: q["relevant"] for q in golden["questions"]}


def run(path):
    """Every saved answer in a JSONL file, checked: (rows, {check: passed of total})."""
    golden = golden_by_id()
    rows, totals = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        results = check(record, golden)
        rows.append({"id": record["id"], "checks": results})
        for name, (passed, _) in results.items():
            done = totals.setdefault(name, [0, 0])
            done[0] += passed
            done[1] += 1
    return rows, totals


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rows, totals = run(argv[0])
    for row in rows:
        failed = {k: why for k, (ok, why) in row["checks"].items() if not ok}
        print(f"{row['id']:4} " + ("ok" if not failed else "; ".join(f"{k}: {why}" for k, why in failed.items())))
    print("\n" + "\n".join(f"  {name}: {passed} of {total}" for name, (passed, total) in totals.items()))
    return 0 if all(p == t for p, t in totals.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
