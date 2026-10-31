"""The owner's check on the judge's marks, and the report that puts it beside the numbers."""

import json

from evals.generation import report as R
from evals.generation import review as V


def claim(verdict="supported", on=True, grounded="cited", cites=(1,)):
    return {"claim": f"{verdict} {on} {grounded}", "quote": "q", "cites": list(cites), "verdict": verdict,
            "source": "", "why": "", "on_question": on, "grounded": grounded, "passage": 0, "ground_why": ""}


def judged():
    fine = [claim() for _ in range(30)]
    return {"Q2": {"claims": [claim("contradicted"), claim(on=False), claim(grounded="none")] + fine[:15],
                   "points made": [1]},
            "Q1": {"claims": fine[15:], "points made": [1]},
            "Q25": {"refused": True}}


def test_every_flagged_claim_is_sampled_and_a_fixed_share_of_the_rest():
    items = V.sample(judged(), size=5)
    flagged = [key for key, *_, kind in items if kind == "flagged"]
    assert flagged == ["Q2#1", "Q2#2", "Q2#3"]
    assert sum(kind == "passed" for *_, kind in items) == 5
    assert items == V.sample(judged(), size=5)                     # the same sample every time
    assert [int(key.split("#")[0][1:]) for key, *_ in items] == sorted(int(k.split("#")[0][1:]) for k, *_ in items)


def test_what_the_judge_said_against_a_claim():
    assert V.flags(claim("contradicted", False, "none", cites=())) == ["contradicted by the key",
                                                                      "backed by no passage", "off the question"]
    # every mark that lowers a number is flagged: a cited passage that does not back the claim lowers citation accuracy
    assert V.flags(claim(grounded="other")) == ["its cited passage does not back it"]
    assert V.flags(claim(grounded="other", cites=())) == []


def test_the_sheet_shows_each_mark_and_where_to_look():
    from evals.generation import claim_judge as J
    records = [{"id": q, "refused": False, "answer": "", "sources": [
        {"n": 1, "file": "02_agent_runtime/runtime.py", "lines": [181, 218], "text": "t"}]} for q in ("Q1", "Q2")]
    text = V.sheet(V.sample(judged(), size=2), records, J.golden_by_id())
    assert "## Q2#1 (flagged)" in text and "against the key: **contradicted**" in text
    assert "the passages it cites: [[1] 02_agent_runtime/runtime.py 181-218]" in text
    assert "the key's evidence: [02_agent_runtime/runtime.py 181-249]" in text


def test_an_existing_review_file_is_never_overwritten(tmp_path, monkeypatch):
    answers = tmp_path / "answers_x.jsonl"
    answers.write_text("\n".join(json.dumps({"id": q, "refused": False, "answer": "", "sources": [
        {"n": 1, "file": "02_agent_runtime/runtime.py", "lines": [181, 218], "text": "t"}]})
        for q in ("Q1", "Q2")) + "\n", encoding="utf-8")
    (tmp_path / "verdicts_x.json").write_text(json.dumps(judged()), encoding="utf-8")
    monkeypatch.setattr(V, "REVIEW", tmp_path / "owner_review.json")
    V.main([str(answers)])
    review = json.loads(V.REVIEW.read_text(encoding="utf-8"))
    assert review["verdicts"] == "verdicts_x.json" and (tmp_path / "review_sheet_x.md").exists()
    review["items"]["Q2#1"].update(right=False, note="mine")
    V.REVIEW.write_text(json.dumps(review), encoding="utf-8")
    V.main([str(answers)])
    assert V.load(V.REVIEW)["items"]["Q2#1"]["note"] == "mine"


def test_accuracy_counts_flagged_and_passed_apart_and_skips_the_unchecked():
    review = {"items": {"a": {"kind": "flagged", "right": True}, "b": {"kind": "flagged", "right": False},
                        "c": {"kind": "passed", "right": True}, "d": {"kind": "passed", "right": None}}}
    assert V.accuracy(review) == {"flagged": (1, 2), "passed": (1, 1)}


def test_the_report_gives_each_answers_numbers_their_means_and_the_owners_check():
    table = [{"id": "Q1", "refused": False, "checks failed": [], "claims": 4, "claims wrong": 1, "faithfulness": 0.75,
              "relevance": 1.0, "completeness": 0.5, "citation accuracy": None},
             {"id": "Q2", "refused": False, "checks failed": ["cites"], "claims": 2, "claims wrong": 0,
              "faithfulness": 0.25, "relevance": 0.5, "completeness": 1.0, "citation accuracy": 1.0},
             {"id": "Q25", "refused": True, "checks failed": ["answered"]}]
    assert R.means(table) == {"faithfulness": 0.5, "relevance": 0.75, "completeness": 0.75,
                              "citation accuracy": 1.0, "claims wrong": 1, "claims": 6, "answers": 2}
    text = R.markdown(table, {"flagged": (3, 4), "passed": (9, 10)}, "answers_x.jsonl")
    assert "| Q1 | - | 4 | 1 | 0.75 | 1.00 | 0.50 | - |" in text
    assert "| Q25 | answered | refused by the floor |" in text
    assert "flagged claims right 3 of 4, passed claims right 9 of 10" in text
    assert "Reviewed by the owner." in R.markdown(table, {"flagged": (3, 4), "passed": (9, 10)}, "x", "the owner")
    assert "Warning" not in text
    table[0]["ground version"], table[1]["ground version"] = 1, 3
    assert "call `ground` under different rules (v1: 1, v3: 1)" in R.markdown(table, None, "x")
    assert "call `check`" not in R.markdown(table, None, "x")
