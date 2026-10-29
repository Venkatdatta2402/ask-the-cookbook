"""The evals' tests run offline, as the app's do: placeholder keys and the saved index with a stand-in embedding
model. Questions are embedded from the vectors kept in evals/data, so nothing is sent anywhere."""

import json

import pytest

import evals.offline  # noqa: F401 -- first: the keys and the stand-in, before anything opens the index


@pytest.fixture(scope="session")
def index():
    return evals.offline.INDEX


@pytest.fixture(scope="session")
def golden():
    from evals.retrieval.evaluate_retrieval import GOLDEN
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def same_index(golden):
    """Whether the saved index is the one the golden set was graded on. The refresh Action can move it on; then the
    exact-span checks and the baseline gate skip, saying so, until the golden set is re-checked and re-baselined."""
    from ask_the_cookbook import answers
    return answers.BUILT["commit"][:7] == golden["index_commit"]
