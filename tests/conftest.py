"""Every test runs offline: placeholder keys, a stand-in embedding model, and fake model clients. No tokens are
spent and no key is needed; the saved index and data/corpus are read as they are."""

import os

os.environ["GROQ_API_KEY"] = "gsk_placeholder_owner"     # set before .env is read: load_dotenv does not override
os.environ["GEMINI_API_KEY"] = "placeholder"
os.environ.pop("GOOGLE_API_KEY", None)                    # Google's library prefers it to GEMINI_API_KEY

import json  # noqa: E402

import pytest  # noqa: E402
from llama_index.core import MockEmbedding  # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

from ask_the_cookbook import search  # noqa: E402

# the index is opened with this when app.py is imported: a question is never sent to Gemini
search.embed_model = lambda: MockEmbedding(embed_dim=search.EMBED_DIM)


@pytest.fixture(scope="session")
def app():
    import app as the_app
    return the_app


@pytest.fixture(scope="session")
def index(app):
    return app.INDEX


@pytest.fixture(autouse=True)
def fresh_day(app):
    app.DAILY.day, app.DAILY.used = None, 0


@pytest.fixture
def ask(app):
    """What the page finally shows for one message: (the reply, the status line). The reply is the chat's new last
    message, or -- when the message was turned away before reaching the chat -- the status line, which says why.
    Pass `session` to continue a chat, and `quote` for a question asked about a selected passage."""
    from ask_the_cookbook.session import Session

    def one(question, key="", mode=app.RAG, session=None, quote=""):
        session = session if session is not None else Session()
        before = len(session.shown)
        shown, _, status, _, _, _ = list(app.respond(question, quote, key, mode, session))[-1]
        return (shown[-1]["content"] if len(shown) > before else status), status
    return one


@pytest.fixture
def fake_client():
    return FakeClient


class FakeClient:
    """An OpenAI client with a script instead of a model: a search first (unless `search` is False), then an
    answer naming the key it was sent on. `fail` is raised on every call instead. Records what it was sent."""

    def __init__(self, key, fail=None, search=True, prompt_tokens=100):
        self.key, self.fail, self.search, self.prompt_tokens = key, fail, search, prompt_tokens
        self.sent = []
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        if self.fail is not None:
            raise self.fail
        messages = kwargs["messages"]
        question = next(m["content"] for m in messages if m.get("role") == "user")
        if self.search and not any(m.get("role") == "tool" for m in messages):
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{self.key}", "type": "function",
                "function": {"name": "search_cookbook", "arguments": json.dumps({"query": question})}}]}
        else:
            message = {"role": "assistant", "content": f"Answered on {self.key} [02_agent_runtime/runtime.py]."}
        return ChatCompletion.model_validate({
            "id": "fake", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 10,
                      "total_tokens": self.prompt_tokens + 10}})
