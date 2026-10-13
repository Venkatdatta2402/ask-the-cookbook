"""Bring the index up to date with the cookbook's latest commit on main.

    python refresh_index.py             count, then refresh: embed new and changed files, remove deleted ones
    python refresh_index.py --dry-run   count only: which files differ and how many chunks they would embed.
                                        Needs no key and writes nothing

Run by hand, or by the GitHub Action, which only runs this. The steps:
    1. the cookbook commit the index was built at, from store/index/cookbook.json
    2. the cookbook's latest commit on main, by `git ls-remote` -- nothing downloaded. Same commit: stop
    3. the cookbook fetched at that commit into a temp folder, and compared FILE BY FILE against the index by
       document hash -- not by git diff, which also flags what nothing indexed sees, such as notebook code cells
    4. corpus/ rewritten as exactly what is indexed, then new and changed files embedded and deleted ones removed
       (ask_index.refresh does both; LlamaIndex's refresh_ref_docs alone never removes a file)
    5. the commit written into store/index/cookbook.json and the README

corpus/ is where the app reads the cookbook's text from -- the agent's open_file, a passage's line numbers: every
indexed file at its path in the cookbook, a notebook cut to its markdown cells. The index is built from it, so the
two cannot disagree.
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "08_frameworks" / "05_llamaindex"))

import ask_index as AI  # noqa: E402
import cookbook as CB  # noqa: E402

URL = "https://github.com/Venkatdatta2402/agentic-forge-cookbook"
CORPUS = ROOT / "corpus"
COMMIT_FILE = AI.INDEX_DIR / "cookbook.json"
README = ROOT / "README.md"
README_MARK = re.compile(r"(<!-- index-commit -->).*?(<!-- /index-commit -->)", re.S)

# Gemini's free tier embeds 1,000 texts a day per Google project, and the day's visitor questions draw on the same
# quota. A change bigger than this is refused before anything is embedded: run it by hand over two days instead.
# A local run saves every 10 files and a second run resumes; the Action's runner keeps nothing between runs.
MAX_EMBEDDINGS = 900


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def recorded():
    """The cookbook commit the index was built at, or None: 05's copied index records none."""
    return json.loads(COMMIT_FILE.read_text(encoding="utf-8"))["commit"] if COMMIT_FILE.exists() else None


def latest():
    return git("ls-remote", URL, "refs/heads/main").split()[0]


def fetch(commit, into):
    """The cookbook at exactly `commit`, fetched by its hash: a push to main after ls-remote changes nothing."""
    git("init", "-q", cwd=into)
    git("fetch", "-q", "--depth", "1", URL, commit, cwd=into)
    git("checkout", "-q", "FETCH_HEAD", cwd=into)


def point_at(repo):
    """cookbook.py reads the repo it sits in (REPO = parents[2]); this points it at another. SELF moves with it:
    it is how 05_llamaindex leaves itself out, and left at this repo's copy the fetched one would be indexed."""
    CB.REPO, CB.SELF = repo, repo / "08_frameworks" / "05_llamaindex"


def compare(index, documents):
    """What a refresh would do, without doing it: the files new to the index, the files whose hash changed, the
    files gone, and how many chunks the new and changed ones split into -- one embedding each."""
    stored = {d.id_: index.docstore.get_document_hash(d.id_) for d in documents}
    new = [d for d in documents if stored[d.id_] is None]
    changed = [d for d in documents if stored[d.id_] not in (None, d.hash)]
    present = {d.id_ for d in documents}
    removed = sorted(ref for ref in index.ref_doc_info if ref not in present)
    chunks = len(AI.split()(new + changed)) if new or changed else 0
    return new, changed, removed, chunks


def snapshot(files, source, to=CORPUS):
    """corpus/ as exactly what is indexed: each file at its path, a notebook as its markdown cells alone.
    Rewritten whole, so a file deleted from the cookbook goes from here too."""
    if to.exists():
        shutil.rmtree(to)
    for path in files:
        out = to / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            prose = [{"cell_type": "markdown", "metadata": {}, "source": c["source"]}
                     for c in cells if c["cell_type"] == "markdown"]
            text = json.dumps({"cells": prose, "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
                              indent=1, ensure_ascii=False) + "\n"
        else:
            text = path.read_text(encoding="utf-8")
        out.write_text(text, encoding="utf-8", newline="\n")


def note(commit, stats):
    """The commit, into the store and the README."""
    COMMIT_FILE.write_text(json.dumps({"commit": commit, "refreshed": date.today().isoformat(),
                                       "files": stats["files"], "chunks": stats["chunks"]}, indent=1) + "\n",
                           encoding="utf-8")
    line = (f"Built from the cookbook at commit [`{commit[:7]}`]({URL}/tree/{commit}): "
            f"{stats['files']} files, {stats['chunks']:,} chunks.")
    text = README.read_text(encoding="utf-8")
    README.write_text(README_MARK.sub(lambda m: f"{m[1]}\n{line}\n{m[2]}", text), encoding="utf-8", newline="\n")


def _writable(func, path, _):
    # git makes its object files read-only, and Windows will not delete a read-only file
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="count what would change; embed and write nothing")
    parser.add_argument("--max-embeddings", type=int, default=MAX_EMBEDDINGS,
                        help=f"refuse a refresh that needs more (default {MAX_EMBEDDINGS})")
    args = parser.parse_args()

    if not README_MARK.search(README.read_text(encoding="utf-8")):
        raise SystemExit("README.md has no <!-- index-commit --> markers to write the commit between")
    built, head = recorded(), latest()
    print(f"index built at {built[:7] if built else '(not recorded)'}; cookbook main at {head[:7]}")
    if built == head and CORPUS.exists():
        print("Up to date: nothing to do.")
        return

    temp = Path(tempfile.mkdtemp(prefix="cookbook-"))
    try:
        fetch(head, temp)
        point_at(temp)
        files, documents = CB.files(), CB.documents()
        # the dry run reads the saved index with a stand-in model: no key, and nothing it could embed
        if args.dry_run:
            from llama_index.core import MockEmbedding
            embed = MockEmbedding(embed_dim=AI.EMBED_DIM)
        else:
            embed = None
        index = AI.open_index(embed=embed)
        new, changed, removed, chunks = compare(index, documents)

        print(f"{len(documents)} files in the cookbook: {len(new)} new, {len(changed)} changed, "
              f"{len(removed)} deleted -> {chunks} chunks to embed")
        for label, group in (("new", [d.id_ for d in new]), ("changed", [d.id_ for d in changed]),
                             ("deleted", removed)):
            for ref in group:
                print(f"  {label:<8}{ref}")
        if args.dry_run:
            return
        if chunks > args.max_embeddings:
            raise SystemExit(f"{chunks} embeddings is over the limit of {args.max_embeddings}. Run it by hand over "
                             f"two days (it resumes), or raise --max-embeddings.")
        snapshot(files, temp)
    finally:
        shutil.rmtree(temp, onexc=_writable)

    point_at(CORPUS)
    from_corpus = CB.documents()
    if [(d.id_, d.hash) for d in from_corpus] != [(d.id_, d.hash) for d in documents]:
        raise SystemExit("corpus/ does not reproduce the cookbook's documents; the index was not touched")
    stats = AI.refresh(index, from_corpus)
    note(head, stats)
    print(f"Done: {stats['files']} files, {stats['chunks']:,} chunks, at {head[:7]}.")


if __name__ == "__main__":
    main()
