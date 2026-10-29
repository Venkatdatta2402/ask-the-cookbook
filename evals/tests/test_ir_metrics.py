"""The metrics, on cases small enough to work out by hand."""

import math

import pytest

from evals.metrics import ir_metrics as M

A, B = "03_tools/invoke.py", "03_tools/README.md"


def test_jaccard_is_shared_lines_over_all_lines():
    assert M.jaccard((1, 10), (1, 10)) == 1
    assert M.jaccard((1, 10), (11, 20)) == 0
    assert M.jaccard((1, 10), (6, 15)) == pytest.approx(5 / 15)
    assert M.jaccard((5, 5), (5, 5)) == 1


def test_a_retrieved_passage_takes_the_grade_of_the_golden_one_it_matches():
    golden = [(A, (1, 86), 3), (A, (89, 154), 2), (B, (77, 84), 1)]
    assert M.gains([(A, (89, 154)), (B, (1, 13)), (A, (1, 86)), (B, (77, 84))], golden) == [2, 0, 3, 1]


def test_each_golden_passage_is_credited_once():
    # two retrieved passages cut from one golden one: the first scores, the second does not
    assert M.gains([(A, (1, 40)), (A, (1, 40))], [(A, (1, 50), 3)]) == [3, 0]


def test_the_file_must_match_and_the_overlap_must_reach_the_threshold():
    golden = [(A, (1, 100), 3)]
    assert M.gains([(B, (1, 100))], golden) == [0]
    assert M.gains([(A, (1, 40))], golden) == [0]                    # 40 of 100 lines: under 0.5
    assert M.gains([(A, (1, 60))], golden) == [3]
    assert M.gains([(A, (1, 40))], golden, min_jaccard=0.4) == [3]


def test_where_one_retrieved_passage_matches_two_the_closer_one_is_taken():
    golden = [(A, (1, 20), 1), (A, (1, 18), 3)]
    assert M.gains([(A, (1, 18))], golden) == [3]


def test_precision_counts_empty_places_as_misses():
    assert M.precision_at_k([3, 0, 1, 0, 2], 5) == pytest.approx(3 / 5)
    assert M.precision_at_k([3], 5) == pytest.approx(1 / 5)
    assert M.precision_at_k([], 5) == 0                              # a question the floor refused


def test_recall_counts_needed_passages_only():
    grades = [3, 3, 2, 1, 1, 1]                                      # three needed
    assert M.recall_at_k([3, 1, 0, 2, 0], grades, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k([1, 1], [1, 1, 1], 5) is None              # nothing needed: no recall to report
    assert M.recall_at_k([3, 3, 2, 0, 0, 3], grades, 5) == 1         # the sixth place is past k


def test_reciprocal_rank_is_one_over_the_first_relevant_place():
    assert M.reciprocal_rank([0, 0, 2, 3]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([1]) == 1
    assert M.reciprocal_rank([0, 0]) == 0


def test_ndcg_is_one_for_the_best_order_and_less_for_any_other():
    grades = [3, 2, 1]
    assert M.ndcg_at_k([3, 2, 1], grades, 5) == pytest.approx(1)
    assert M.ndcg_at_k([1, 2, 3], grades, 5) < 1
    assert M.ndcg_at_k([], grades, 5) == 0
    assert M.ndcg_at_k([1], [], 5) == 0


def test_ndcg_by_hand():
    # DCG of [0, 3]: 7 / log2(3). Ideal [3]: 7 / log2(2) = 7
    assert M.ndcg_at_k([0, 3], [3], 5) == pytest.approx(1 / math.log2(3))


def test_ndcg_only_counts_the_first_k_places():
    assert M.ndcg_at_k([0, 0, 3], [3], 2) == 0


def test_score_puts_it_together():
    out = M.score([(A, (1, 86)), (B, (1, 13))], [(A, (1, 86), 3), (B, (77, 84), 2)], k=2)
    assert out == {"gains": [3, 0], "precision": 0.5, "recall": 0.5, "mrr": 1.0,
                   "ndcg": pytest.approx(7 / (7 + 3 / math.log2(3)))}


def test_mean_skips_questions_without_the_metric():
    assert M.mean([{"recall": 1.0}, {"recall": None}, {"recall": 0.5}], "recall") == pytest.approx(0.75)
