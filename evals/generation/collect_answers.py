"""SPENDS GROQ TOKENS. The app's RAG answers to the golden questions, saved for the checks and the judge.

Run by hand, never in CI. Each question is answered as the app answers it -- the same engine, model and prompt --
from the same passages the retrieval eval measures: the search is given the question's kept embedding
(data/query_embeddings.json), so no Gemini call is made and the passages are exactly the ones scored there. If the
floor lets a question through, answering it is about 3,000-4,000 Groq tokens (measured live: 2,810 and 4,027), so
the 24 RAG questions come to roughly 90,000 of Groq's free 200,000 a day. Only RAG questions: an agentic answer is
30,000-40,000 tokens and is evaluated separately. Answers are appended as they come, so a run stopped by a limit
carries on where it stopped.

    python -m evals.generation.collect_answers [Q1 Q3 ...]      key from .env: GROQ_API_KEY
"""

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GOLDEN = HERE / "data" / "retrieval_golden.json"
EMBEDDINGS = HERE / "data" / "query_embeddings.json"
OUT = HERE / "reports" / f"answers_{date.today().isoformat()}.jsonl"


def answer(question, vector, llm):
    """`answers.rag`, with the question's kept embedding: (refused, the model's text, the response or None)."""
    from llama_index.core import QueryBundle

    from ask_the_cookbook import answers, search
    engine = search.engine(answers.INDEX, llm)
    nodes = engine.retrieve(QueryBundle(query_str=question, embedding=vector))
    if not nodes:
        return True, search.NOT_COVERED, None
    response = engine.synthesize(QueryBundle(question), nodes)
    return not response.source_nodes, str(response).strip(), response


def record(question_id, question, refused, text, response, tokens=None):
    """One answer as the checks and the judge read it: the model's own text, and the numbered sources it was given --
    each with its text as the model was shown it (CitationQueryEngine's "Source n:" piece, with the metadata the
    model sees), which is what the judge grounds the claims in."""
    from llama_index.core.schema import MetadataMode

    from ask_the_cookbook import answers
    sources = []
    if response is not None:
        for n, s in enumerate(response.source_nodes, 1):
            lines = answers.span(s.node)
            sources.append({"n": n, "file": s.node.metadata["file"], "lines": list(lines) if lines else [0, -1],
                            "text": s.node.get_content(metadata_mode=MetadataMode.LLM)})
    return {"id": question_id, "question": question, "refused": refused, "answer": text, "sources": sources,
            "tokens": tokens, "index": answers.BUILT["commit"][:7]}


def main(argv):
    import os

    from ask_the_cookbook import answers
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    vectors = json.loads(EMBEDDINGS.read_text(encoding="utf-8"))["vectors"]
    wanted = set(argv) or {q["id"] for q in golden["questions"] if q["kind"] == "rag"}
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if OUT.exists() else set()
    for q in golden["questions"]:
        if q["id"] not in wanted or q["id"] in done:
            continue
        llm = answers.chat_model(os.environ["GROQ_API_KEY"])
        refused, text, response = answer(q["question"], vectors[q["question"]], llm)
        with OUT.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record(q["id"], q["question"], refused, text, response, llm.budget.spent),
                                 ensure_ascii=False) + "\n")
        print(f"{q['id']}: {'refused' if refused else 'answered'}, {llm.budget.spent:,} tokens")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
