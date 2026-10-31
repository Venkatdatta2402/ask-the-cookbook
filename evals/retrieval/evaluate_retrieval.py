"""The app's retrieval, measured against the golden set -- offline, deterministic, and cheap enough for every push.

The search measured is the one RAG answers from: `search.CookbookRetriever` (vector search and BM25, fused by
reciprocal rank, behind the floor), on the saved index, k=5 as `search.engine` uses it. Each question's embedding is
the one kept in data/query_embeddings.json, so no Gemini call is made.

All 30 questions are run, the agentic ones too: an agent's searches are its own rewordings, but how well one
search of the whole question does is still worth knowing. Means are reported over all 30, the 24 RAG questions,
and the 6 agentic ones.

    python -m evals.retrieval.evaluate_retrieval              the report, to reports/retrieval_latest.json
    python -m evals.retrieval.evaluate_retrieval --baseline   the same, also written as the committed baseline
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GOLDEN = HERE / "data" / "retrieval_golden.json"
EMBEDDINGS = HERE / "data" / "query_embeddings.json"
REPORTS = HERE / "reports"
BASELINE = REPORTS / "retrieval_baseline.json"
LATEST = REPORTS / "retrieval_latest.json"

K = 5
METRICS = ("precision", "recall", "mrr", "ndcg")
# how far a mean may fall below the baseline before the gate fails. The search is deterministic, so on the same
# index and code every number repeats exactly; this only leaves room for float noise across machines
TOLERANCE = 0.005


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve(index, question, vector, k=K):
    """([(file, (a, b))] the search returns for one question, best first -- [] when the floor refuses it; and the
    vector search's best raw score, the number the floor is applied to)."""
    from llama_index.core import QueryBundle
    from llama_index.core.llms import MockLLM

    from ask_the_cookbook import answers, search
    retriever = search.CookbookRetriever(index, MockLLM(), k=k)
    found = retriever.retrieve(QueryBundle(query_str=question, embedding=vector))
    return [(s.node.metadata["file"], tuple(answers.span(s.node) or (0, -1))) for s in found], retriever.best


def evaluate(index=None, k=K):
    """The report: every question's retrieved passages, gains and metrics, and the means."""
    from evals.metrics import ir_metrics as M
    if index is None:
        from evals.offline import INDEX as index
    golden, vectors = load(GOLDEN), load(EMBEDDINGS)["vectors"]
    rows = []
    for q in golden["questions"]:
        retrieved, best = retrieve(index, q["question"], vectors[q["question"]], k)
        graded = [(r["file"], tuple(r["lines"]), r["grade"]) for r in q["relevant"]]
        rows.append({"id": q["id"], "kind": q["kind"], "best_score": round(best, 4) if best is not None else None,
                     "retrieved": [{"file": f, "lines": list(l)} for f, l in retrieved],
                     **M.score(retrieved, graded, k)})
    means = {}
    for name, subset in (("all", rows), ("rag", [r for r in rows if r["kind"] == "rag"]),
                         ("agentic", [r for r in rows if r["kind"] == "agentic"])):
        # recall_best is not gated: it depends on the golden set alone, not on the search
        means[name] = {m: round(M.mean(subset, m), 4) for m in (*METRICS, "recall_best")} | {"questions": len(subset)}
    return {"k": k, "index_commit": golden["index_commit"], "means": means, "questions": rows}


def regressions(report, baseline, tolerance=TOLERANCE):
    """Each mean that fell below the baseline by more than `tolerance`, as a sentence. [] when nothing did."""
    out = []
    for subset, base in baseline["means"].items():
        for metric in METRICS:
            now = report["means"][subset][metric]
            if now < base[metric] - tolerance:
                out.append(f"{subset} {metric}: {now:.4f}, baseline {base[metric]:.4f}")
    return out


def table(report):
    """The means and every question's line, for reading."""
    lines = [f"k={report['k']}, index {report['index_commit']}"]
    for subset, m in report["means"].items():
        lines.append(f"  {subset:8} ({m['questions']:2})  P@{report['k']} {m['precision']:.3f}  "
                     f"R@{report['k']} {m['recall']:.3f} (best possible {m['recall_best']:.3f})  MRR {m['mrr']:.3f}  NDCG@{report['k']} {m['ndcg']:.3f}")
    for r in report["questions"]:
        recall = "  -  " if r["recall"] is None else f"{r['recall']:.2f}"
        lines.append(f"  {r['id']:4} {r['kind']:8} gains {r['gains']}  P {r['precision']:.2f}  R {recall}  "
                     f"MRR {r['mrr']:.2f}  NDCG {r['ndcg']:.2f}")
    return "\n".join(lines)


def main(argv):
    report = evaluate()
    REPORTS.mkdir(exist_ok=True)
    text = json.dumps(report, indent=1) + "\n"
    LATEST.write_text(text, encoding="utf-8")
    if "--baseline" in argv:
        BASELINE.write_text(text, encoding="utf-8")
    print(table(report))
    if BASELINE.exists() and "--baseline" not in argv:
        fell = regressions(report, load(BASELINE))
        print("\nagainst the baseline: " + ("; ".join(fell) if fell else "nothing fell"))
        return 1 if fell else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
