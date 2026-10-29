"""The answer checks, on answers written to pass or fail them."""

import pytest

from evals.generation import deterministic_checks as D


@pytest.fixture(scope="module")
def golden():
    return D.golden_by_id()


def answer(text, sources=None, refused=False, qid="Q2"):
    return {"id": qid, "question": "", "refused": refused, "answer": text,
            "sources": sources if sources is not None else [{"n": 1, "file": "02_agent_runtime/runtime.py",
                                                              "lines": [181, 218]}]}


def failed(results):
    return {name for name, (ok, _) in results.items() if not ok}


def test_a_good_answer_passes_everything(golden):
    results = D.check(answer("A doom loop is the same call twice [1]."), golden)
    assert failed(results) == set()
    assert set(results) == {"answered", "cites", "citations resolve", "sources exist", "cites an answer passage"}


def test_a_citation_with_no_source_behind_it_fails(golden):
    assert "citations resolve" in failed(D.check(answer("See [1] and [3]."), golden))


def test_an_answer_that_cites_nothing_fails(golden):
    assert failed(D.check(answer("A doom loop is the same call twice."), golden)) >= {"cites",
                                                                                      "cites an answer passage"}


def test_every_citation_form_the_model_writes_is_read():
    assert D.cited("one [1], two [Source 2], three 【3†L1-L4】, four [1, 4]") == [1, 2, 3, 1, 4]


def test_brackets_in_code_are_not_citations():
    assert D.cited("`pairs[0] == pairs[2]` and ```\nx[1]\n``` but [2]") == [2]


def test_a_source_that_is_not_real_lines_of_a_real_file_fails(golden):
    made_up = [{"n": 1, "file": "02_agent_runtime/nowhere.py", "lines": [1, 5]}]
    too_long = [{"n": 1, "file": "02_agent_runtime/runtime.py", "lines": [181, 99999]}]
    assert "sources exist" in failed(D.check(answer("[1]", made_up), golden))
    assert "sources exist" in failed(D.check(answer("[1]", too_long), golden))


def test_citing_only_passages_that_do_not_answer_the_question_is_flagged(golden):
    elsewhere = [{"n": 1, "file": "03_tools/README.md", "lines": [1, 13]}]
    assert "cites an answer passage" in failed(D.check(answer("[1]", elsewhere), golden))


def test_a_listed_but_uncited_answer_passage_does_not_count(golden):
    sources = [{"n": 1, "file": "03_tools/README.md", "lines": [1, 13]},
               {"n": 2, "file": "02_agent_runtime/runtime.py", "lines": [181, 218]}]
    assert "cites an answer passage" in failed(D.check(answer("[1]", sources), golden))


def test_refusing_a_covered_question_fails_and_refusing_an_unknown_one_passes(golden):
    assert failed(D.check(answer("", [], refused=True), golden)) == {"answered"}
    assert failed(D.check(answer("", [], refused=True, qid="off-topic"), golden)) == set()


def test_an_empty_answer_fails(golden):
    assert "answered" in failed(D.check(answer("  ", []), golden))


def test_a_file_of_answers_is_totalled(tmp_path):
    import json
    path = tmp_path / "answers.jsonl"
    path.write_text("\n".join(json.dumps(a) for a in (answer("[1]"), answer("[2]"))) + "\n", encoding="utf-8")
    rows, totals = D.run(path)
    assert [r["id"] for r in rows] == ["Q2", "Q2"]
    assert totals["citations resolve"] == [1, 2]
