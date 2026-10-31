"""The golden set is what every retrieval number rests on, so it is checked against the index itself: every graded
passage is an indexed chunk, and every quote is that chunk's own text."""

import json

import pytest

from evals.retrieval.evaluate_retrieval import EMBEDDINGS


def test_thirty_questions_in_order(golden):
    assert [q["id"] for q in golden["questions"]] == [f"Q{n}" for n in range(1, 31)]
    assert {q["kind"] for q in golden["questions"]} == {"rag", "agentic"}


def test_every_entry_is_graded_and_explained(golden):
    for q in golden["questions"]:
        for r in q["relevant"]:
            assert r["grade"] in (1, 2, 3), (q["id"], r)
            assert r["quote"].strip() and r["why"].strip(), (q["id"], r)


def test_no_passage_is_graded_twice_for_one_question(golden):
    for q in golden["questions"]:
        spans = [(r["file"], tuple(r["lines"])) for r in q["relevant"]]
        assert len(spans) == len(set(spans)), q["id"]


def test_every_question_has_a_passage_that_answers_it(golden):
    for q in golden["questions"]:
        assert any(r["grade"] == 3 for r in q["relevant"]), q["id"]


def test_every_graded_passage_is_an_indexed_chunk_and_its_quote_is_in_it(golden, index, same_index):
    if not same_index:
        pytest.skip(f"the index is no longer the one graded ({golden['index_commit']}): re-check the golden set")
    from ask_the_cookbook import answers
    chunks = {}
    for node in index.docstore.docs.values():
        lines = answers.span(node)
        if lines:
            chunks.setdefault((node.metadata["file"], tuple(lines)), []).append(node.get_content())
    assert len(index.docstore.docs) == golden["chunks"]
    for q in golden["questions"]:
        for r in q["relevant"]:
            texts = chunks.get((r["file"], tuple(r["lines"])))
            assert texts, f"{q['id']}: {r['file']} {r['lines']} is not an indexed chunk"
            assert any(r["quote"] in t for t in texts), f"{q['id']}: quote not in {r['file']} {r['lines']}"


def test_every_rag_question_has_reference_points_for_the_judge(golden):
    from evals.generation.claim_judge import golden_by_id
    points = golden_by_id()
    for q in golden["questions"]:
        if q["kind"] == "rag":
            assert points[q["id"]]["question"] == q["question"] and points[q["id"]]["points"], q["id"]
    for q in points.values():
        assert all(p["point"].strip() and p["quotes"] for p in q["points"]), q["id"]


def test_every_reference_quote_is_in_its_evidence_lines(same_index, golden):
    if not same_index:
        pytest.skip(f"the index is no longer the one graded ({golden['index_commit']}): re-check the reference points")
    import re

    from ask_the_cookbook import answers, search
    from evals.generation.claim_judge import golden_by_id

    def flat(text):
        # a sentence can run over lines, and a code comment's lines each start with "#"
        return " ".join(" ".join(re.sub(r"^\s*#\s?", "", line) for line in text.splitlines()).split())

    for q in golden_by_id().values():
        shown = []
        for file, first, last in q["evidence"]:
            lines = search.file_text(file).splitlines() if file in answers.FILES else []
            assert 1 <= first <= last <= len(lines), (q["id"], file, first, last)
            shown.append(flat("\n".join(lines[first - 1:last])))
        for i, p in enumerate(q["points"], 1):
            for quote in p["quotes"]:
                assert flat(quote) in " ".join(shown), f"{q['id']} point {i}: {quote[:60]}"


def test_every_question_has_its_kept_embedding(golden):
    kept = json.loads(EMBEDDINGS.read_text(encoding="utf-8"))
    assert kept["model"] == "gemini-embedding-001" and kept["task"] == "RETRIEVAL_QUERY"
    for q in golden["questions"]:
        assert len(kept["vectors"][q["question"]]) == kept["dims"] == 768
