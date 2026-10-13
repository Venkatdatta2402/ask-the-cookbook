"""data/corpus and the saved index describe the same cookbook, and the corpus rules leave out what they should."""

import json

from ask_the_cookbook import corpus as CB
from ask_the_cookbook import search


def test_index_and_corpus_hold_the_same_files_at_the_same_hashes(index):
    documents = CB.documents()
    assert {d.id_ for d in documents} == set(index.ref_doc_info)
    assert all(index.docstore.get_document_hash(d.id_) == d.hash for d in documents)


def test_the_recorded_counts_match(index):
    built = json.loads((search.INDEX_DIR / "cookbook.json").read_text(encoding="utf-8"))
    assert built["files"] == len(index.ref_doc_info)
    assert built["chunks"] == len(index.docstore.docs)


def test_left_out_the_assistant_its_evaluators_tests_and_stubs():
    files = [p.relative_to(CB.REPO).as_posix() for p in CB.files()]
    assert files[0] == "README.md"
    assert not [f for f in files if f.startswith(("08_frameworks/05_llamaindex/", "08_frameworks/07_langsmith/",
                                                  "08_frameworks/08_langfuse/"))]
    assert not [f for f in files if f.rsplit("/", 1)[-1].startswith("test_") or f.endswith("_stubs.py")]


def test_the_corpus_notebooks_hold_markdown_cells_only():
    notebooks = [p for p in CB.files() if p.suffix == ".ipynb"]
    assert notebooks
    for path in notebooks:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        assert {c["cell_type"] for c in cells} <= {"markdown"}, path
