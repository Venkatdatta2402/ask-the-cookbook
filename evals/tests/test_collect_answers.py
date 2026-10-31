"""Collecting answers, offline: a stand-in model, and the question's kept embedding in place of Gemini."""

import json

from llama_index.core.llms import MockLLM

from evals.generation import collect_answers as C
from evals.retrieval import evaluate_retrieval as E


def test_an_answer_is_written_from_the_passages_the_retrieval_eval_scores(index, golden):
    vectors = json.loads(E.EMBEDDINGS.read_text(encoding="utf-8"))["vectors"]
    q = next(q for q in golden["questions"] if q["id"] == "Q2")
    refused, text, response = C.answer(q["question"], vectors[q["question"]], MockLLM())
    saved = C.record("Q2", q["question"], refused, text, response, tokens=0)
    scored, _ = E.retrieve(index, q["question"], vectors[q["question"]])
    assert not refused and [(s["file"], tuple(s["lines"])) for s in saved["sources"]] == scored
    assert [s["n"] for s in saved["sources"]] == list(range(1, len(scored) + 1))
    # each passage kept as the model was shown it, for the judge to ground the claims in
    assert all(s["text"].startswith(f"Source {s['n']}:") or f"Source {s['n']}:" in s["text"] for s in saved["sources"])


def test_a_question_the_floor_refuses_is_saved_as_refused(golden):
    vectors = json.loads(E.EMBEDDINGS.read_text(encoding="utf-8"))["vectors"]
    q = next(q for q in golden["questions"] if q["id"] == "Q25")         # below the floor: 0.613 < 0.66
    refused, _, response = C.answer(q["question"], vectors[q["question"]], MockLLM())
    assert refused and response is None
    assert C.record("Q25", q["question"], refused, "", response)["sources"] == []
