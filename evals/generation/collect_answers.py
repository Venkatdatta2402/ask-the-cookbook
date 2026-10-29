"""SPENDS TOKENS. The app's RAG answers to the golden questions, saved for the checks and, later, a judge.

Run by hand, never in CI: each question is one Gemini embedding and, if the floor lets it through, about 3,000-4,000
Groq tokens (measured live: 2,810 and 4,027) -- the 24 RAG questions come to roughly 90,000, close to half of
Groq's free 200,000 a day. Only RAG questions: an agentic answer is 30,000-40,000 tokens and is evaluated
separately. Answers are appended as they come, so a run stopped by a limit carries on where it stopped.

    python -m evals.generation.collect_answers [Q1 Q3 ...]      keys from .env: GROQ_API_KEY, GEMINI_API_KEY
"""

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GOLDEN = HERE / "data" / "retrieval_golden.json"
OUT = HERE / "reports" / f"answers_{date.today().isoformat()}.jsonl"


def record(question_id, question, result, answers):
    """One answer as deterministic_checks reads it: the model's own text and the numbered sources it was given."""
    sources = []
    if result.get("response") is not None:
        for n, s in enumerate(result["response"].source_nodes, 1):
            lines = answers.span(s.node)
            sources.append({"n": n, "file": s.node.metadata["file"], "lines": list(lines) if lines else [0, -1]})
    return {"id": question_id, "question": question, "refused": result["refused"], "answer": result["answer"],
            "sources": sources}


def main(argv):
    import os

    from ask_the_cookbook import answers
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    wanted = set(argv) or {q["id"] for q in golden["questions"] if q["kind"] == "rag"}
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if OUT.exists() else set()
    for q in golden["questions"]:
        if q["id"] not in wanted or q["id"] in done:
            continue
        result = answers.rag(q["question"], os.environ["GROQ_API_KEY"], {})
        with OUT.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record(q["id"], q["question"], result, answers), ensure_ascii=False) + "\n")
        print(f"{q['id']}: {'refused' if result['refused'] else 'answered'}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
