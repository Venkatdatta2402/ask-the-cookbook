"""The gate: retrieval on the golden set must not fall below the committed baseline. A change to the search, the
splitter or the floor that costs relevant passages fails here, on every push, for nothing."""

import json

import pytest

from evals.retrieval import evaluate_retrieval as E


@pytest.fixture(scope="module")
def report(index):
    return E.evaluate(index)


def test_retrieval_holds_its_baseline(report, same_index):
    if not same_index:
        pytest.skip("the index has moved on from the one graded: re-check the golden set, then re-baseline")
    baseline = json.loads(E.BASELINE.read_text(encoding="utf-8"))
    fell = E.regressions(report, baseline)
    assert not fell, "retrieval fell below its baseline: " + "; ".join(fell)


def test_the_same_search_gives_the_same_numbers(report, index):
    again = E.evaluate(index)
    assert [q["gains"] for q in again["questions"]] == [q["gains"] for q in report["questions"]]
    assert again["means"] == report["means"]


def test_a_fall_is_reported_and_a_rise_is_not():
    base = {"means": {"all": {"precision": 0.8, "recall": 0.4, "mrr": 0.9, "ndcg": 0.7}}}
    worse = {"means": {"all": {"precision": 0.7, "recall": 0.4, "mrr": 0.95, "ndcg": 0.7}}}
    assert E.regressions(worse, base) == ["all precision: 0.7000, baseline 0.8000"]
    assert E.regressions(base, base) == []


def test_every_question_is_scored(report):
    assert len(report["questions"]) == 30
    assert report["means"]["all"]["questions"] == 30
    for q in report["questions"]:
        assert len(q["gains"]) == len(q["retrieved"]) <= E.K
