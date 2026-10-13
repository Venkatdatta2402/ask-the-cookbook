"""The corpus: the cookbook, as the documents an "ask the cookbook" assistant answers from.

    REPO           the cookbook to read: data/corpus (refresh.py writes it), or a fetched copy (refresh.py points here)
    documents()    every README, the prose of every notebook, and every module, as LlamaIndex `Document`s

What goes in, and what stays out:
    in     READMEs; notebook MARKDOWN cells, which is where the notebooks explain; .py modules, which is
           where the explained things are implemented
    out    notebook code cells and outputs -- the code is mostly calls into the modules, which are indexed
           themselves, and the outputs are logs and tables; tests and stand-ins; stored runs; and this
           folder and the two that evaluate it (07_langsmith, 08_langfuse), so the assistant cannot find its
           own answer key
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "data" / "corpus"
# the cookbook's folder for this assistant, which it leaves out: it holds the assistant's own code and eval questions
SELF = ("08_frameworks", "05_llamaindex")

_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "store", ".git"}
# the projects that evaluate this assistant: 07_langsmith holds its answer key (eval_set.py, hand_labels.py), and an
# assistant that can retrieve its own answer key is not being tested
_EVALUATORS = {"07_langsmith", "08_langfuse"}


def _skipped(path):
    rel = path.relative_to(REPO).parts
    parts = set(rel)
    return (bool(parts & _SKIP_DIRS) or rel[:len(SELF)] == SELF or bool(parts & _EVALUATORS)
            or path.name.startswith("test_") or path.name.endswith("_stubs.py"))


def files():
    """Every file the assistant reads, in a stable order."""
    found = [REPO / "README.md"]
    for chapter in sorted(p for p in REPO.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        for pattern in ("**/README.md", "**/*.ipynb", "**/*.py"):
            found += sorted(p for p in chapter.glob(pattern) if not _skipped(p))
    return found


def _notebook_prose(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")


def documents():
    """One `Document` per file. Its id is the file's path, so a changed file replaces its old version."""
    from llama_index.core import Document
    docs = []
    for path in files():
        rel = path.relative_to(REPO).as_posix()
        kind = "notebook" if path.suffix == ".ipynb" else "code" if path.suffix == ".py" else "readme"
        text = _notebook_prose(path) if kind == "notebook" else path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chapter = rel.split("/")[0] if "/" in rel else "(repo)"
        docs.append(Document(id_=rel, text=text, metadata={"file": rel, "chapter": chapter, "kind": kind}))
    return docs
