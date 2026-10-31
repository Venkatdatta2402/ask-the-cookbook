"""The generation report, from what is saved -- no model call: every RAG answer's free checks and the judge's five
numbers, their means, and how many of the judge's marks the owner found right.

    python -m evals.generation.report <answers.jsonl>      the report, printed and written to reports/generation_<date>.md
"""

import sys
from pathlib import Path

NUMBERS = ("faithfulness", "relevance", "completeness", "citation accuracy")


def rows(records, checks, judged, golden):
    """One row per answer: the free checks that failed, and the judge's numbers."""
    from evals.generation import claim_judge as J
    out = []
    for r in records:
        found = judged.get(r["id"], {})
        row = {"id": r["id"], "refused": r["refused"],
               "checks failed": [name for name, (ok, _) in checks[r["id"]].items() if not ok]}
        if "claims" in found:
            row.update(J.scores(found, golden[r["id"]]), **{"ground version": found.get("ground version", 1),
                                                            "check version": found.get("check version", 1)})
        out.append(row)
    return out


def means(table):
    """Each number's mean over the answers that have it, and the claims wrong in all."""
    judged = [r for r in table if "claims" in r]
    out = {}
    for name in NUMBERS:
        values = [r[name] for r in judged if r[name] is not None]
        out[name] = round(sum(values) / len(values), 3) if values else None
    out["claims wrong"] = sum(r["claims wrong"] for r in judged)
    out["claims"] = sum(r["claims"] for r in judged)
    out["answers"] = len(judged)
    return out


def markdown(table, review, answers_name, reviewer=None):
    show = lambda v: "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = [f"# Generation report: {answers_name}", "",
             "| | free checks failed | claims | wrong | faithfulness | relevance | completeness | citation accuracy |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        failed = ", ".join(r["checks failed"]) or "-"
        if r["refused"]:
            lines.append(f"| {r['id']} | {failed} | refused by the floor | | | | | |")
        elif "claims" not in r:
            lines.append(f"| {r['id']} | {failed} | not judged yet | | | | | |")
        else:
            lines.append(f"| {r['id']} | {failed} | {r['claims']} | {r['claims wrong']} | " +
                         " | ".join(show(r[name]) for name in NUMBERS) + " |")
    m = means(table)
    for call, numbers in (("ground", "faithfulness and citation accuracy"), ("check", "claims wrong and relevance")):
        versions = sorted({r[f"{call} version"] for r in table if f"{call} version" in r})
        if len(versions) > 1:
            # answers judged under different rules for a call do not average
            lines += ["", f"**Warning:** these answers went through call `{call}` under different rules (" + ", ".join(
                f"v{v}: {sum(r.get(f'{call} version') == v for r in table)}" for v in versions) +
                f"): {numbers} mix them. Run the judge with `--rejudge`."]
    if m["answers"]:
        lines += ["", f"**Over {m['answers']} answers:** " + ", ".join(f"{name} {show(m[name])}" for name in NUMBERS) +
                  f"; {m['claims wrong']} claims wrong of {m['claims']}."]
    if review:
        (fr, ft), (pr, pt) = review["flagged"], review["passed"]
        if ft or pt:
            lines += ["", f"**The check on the judge's marks:** flagged claims right {fr} of {ft}, "
                          f"passed claims right {pr} of {pt}." + (f" Reviewed by {reviewer}." if reviewer else "")]
    return "\n".join(lines) + "\n"


def main(argv):
    import json

    from evals.generation import claim_judge as J
    from evals.generation import deterministic_checks as D
    from evals.generation import review as V
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    records = J.load_answers(path)
    checks = {r["id"]: D.check(r, D.golden_by_id()) for r in records}
    judged = json.loads(J.verdicts_path(path).read_text(encoding="utf-8")) if J.verdicts_path(path).exists() else {}
    checked = V.load() if V.REVIEW.exists() else None
    review = V.accuracy(checked) if checked else None
    text = markdown(rows(records, checks, judged, J.golden_by_id()), review, path.name,
                    reviewer=(checked or {}).get("reviewed by"))
    out = path.with_name(path.stem.replace("answers", "generation", 1) + ".md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
