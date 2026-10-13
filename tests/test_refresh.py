"""The refresh's offline parts: what it would embed, and the corpus snapshot it writes. Fetching the cookbook and
embedding are the parts left out -- the one needs the network, the other a key and quota."""

from llama_index.core import Document

from ask_the_cookbook import corpus as CB
from ask_the_cookbook import refresh


def test_an_up_to_date_index_has_nothing_to_embed(index):
    new, changed, removed, chunks = refresh.compare(index, CB.documents())
    assert (new, changed, removed, chunks) == ([], [], [], 0)


def test_a_changed_new_and_deleted_file_are_each_seen(index):
    documents = CB.documents()
    first, rest = documents[0], documents[2:]       # documents[1] left out: deleted
    edited = Document(id_=first.id_, text=first.text + "\n\nOne more paragraph.", metadata=first.metadata)
    added = Document(id_="09_new/README.md", text="# A new chapter\n\nAbout something new.",
                     metadata={"file": "09_new/README.md", "chapter": "09_new", "kind": "readme"})
    new, changed, removed, chunks = refresh.compare(index, [edited, added, *rest])
    assert [d.id_ for d in new] == ["09_new/README.md"]
    assert [d.id_ for d in changed] == [first.id_]
    assert removed == [documents[1].id_]
    assert chunks >= 2


def test_the_snapshot_reproduces_the_documents(tmp_path, monkeypatch):
    before = [(d.id_, d.hash) for d in CB.documents()]
    refresh.snapshot(CB.files(), CB.REPO, to=tmp_path / "corpus")
    monkeypatch.setattr(CB, "REPO", tmp_path / "corpus")
    assert [(d.id_, d.hash) for d in CB.documents()] == before
