"""The API: keys checked before any work, answers streamed as server-sent events (or one JSON reply), the limits,
requests worked out at once, and a caller who leaves stopping the work. Offline: stand-in models, fake clients, and
the key check faked -- a key is "refused" when it has "bad" in it."""

import asyncio
import json
import time

import httpx
import openai
import pytest
from fastapi import FastAPI
from llama_index.core.llms import MockLLM
from openai.types.chat import ChatCompletion

from ask_the_cookbook import answers, api
from conftest import EMBEDDED_FOR

SERVER = FastAPI(routes=api.router.routes)      # the API's routes, as app.py hands them to the app Gradio makes
GROQ = {"X-Groq-Key": "gsk_visitor"}
BOTH = {"X-Groq-Key": "gsk_visitor", "X-Gemini-Key": "AIza-caller"}
QUESTION = "What counts as a doom loop?"


@pytest.fixture(autouse=True)
def fresh_api(monkeypatch):
    """Limits and slots of their own for each test, and no key check reaching Groq or Gemini."""
    async def check_key(kind, key):
        return "bad" not in key
    monkeypatch.setattr(api, "check_key", check_key)
    monkeypatch.setattr(api, "LIMIT", api.PerMinute(api.PER_MINUTE))
    monkeypatch.setattr(api, "GATES", api.Gates(api.RUNS_AT_ONCE))
    EMBEDDED_FOR.clear()


@pytest.fixture
def offline_rag(monkeypatch):
    """The real async RAG path on a stand-in LLM, and no floor: a stand-in embedding is near nothing, so with the
    floor every question would be turned away before the sources could be seen."""
    monkeypatch.setattr(answers, "chat_model", lambda key: MockLLM(max_tokens=20))
    monkeypatch.setattr(answers.A, "FLOOR", -1)


def send(*requests):
    """Requests to the API, all at once in one event loop: [(method, path, kwargs), ...] -> responses."""
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=SERVER), base_url="http://test",
                                     timeout=30) as client:
            return await asyncio.gather(*(client.request(m, p, **kw) for m, p, kw in requests))
    return asyncio.run(go())


def ask(body, headers=GROQ, address="203.0.113.1"):
    return send(("POST", "/api/ask", {"json": body, "headers": {**headers, "X-Forwarded-For": address}}))[0]


def events_of(stream):
    """A server-sent events body as [(event, data), ...]; comment lines (keep-alives) left out."""
    out = []
    for block in stream.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if lines:
            out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


# --------------------------------------------------------------------------- checked before any work

@pytest.mark.parametrize("body, headers, status, says", [
    ({"question": QUESTION}, {}, 401, "X-Groq-Key"),
    ({"question": QUESTION}, {"X-Groq-Key": "sk-not-groq"}, 401, "X-Groq-Key"),
    ({"question": QUESTION, "mode": "agentic"}, GROQ, 401, "X-Gemini-Key"),
    ({"question": QUESTION}, {"X-Groq-Key": "gsk_bad"}, 401, "That Groq key was refused"),
    ({"question": QUESTION, "mode": "agentic"}, {**GROQ, "X-Gemini-Key": "bad"}, 401, "That Gemini key was refused"),
])
def test_keys_are_checked_before_any_work(monkeypatch, body, headers, status, says):
    async def never(*args, **kwargs):
        raise AssertionError("work began before the keys were checked")
    monkeypatch.setattr(answers, "arag", never)
    monkeypatch.setattr(answers, "agent_parts", never)
    reply = ask(body, headers)
    assert reply.status_code == status and says in reply.json()["detail"]
    assert EMBEDDED_FOR == []                   # no search was made: no embedding on anyone's key


@pytest.mark.parametrize("question", ["   ", "x" * (answers.MAX_QUESTION + 1)])
def test_an_empty_or_overlong_question_is_rejected(question):
    assert ask({"question": question}).status_code == 422


# --------------------------------------------------------------------------- RAG

def test_rag_streams_what_is_happening_then_the_answer_with_linked_sources(offline_rag):
    reply = ask({"question": QUESTION})
    assert reply.headers["content-type"].startswith("text/event-stream")
    events = events_of(reply.text)
    names = [name for name, _ in events]
    assert names[:2] == ["progress", "progress"] and names[-2:] == ["answer", "done"]
    assert events[0][1]["message"] == "Searching the cookbook"
    assert events[1][1]["message"].startswith("Reading 5 passages")
    answer = events[-2][1]
    assert answer["sources"] and all(f"/blob/{answers.BUILT['commit']}/" in s["url"] for s in answer["sources"])
    assert answer["refused"] is False and answer["tool_calls"] == []


def test_rag_as_one_json_reply(offline_rag):
    reply = ask({"question": QUESTION, "stream": False})
    assert reply.status_code == 200 and reply.headers["content-type"].startswith("application/json")
    body = reply.json()
    assert body["sources"] and body["seconds"] >= 0 and not body["refused"]


def test_a_question_nothing_is_near_is_refused_without_a_model_call(monkeypatch):
    from llama_index.core.query_engine import CitationQueryEngine

    async def never(*args, **kwargs):
        raise AssertionError("the answer was written for a question the floor turned away")
    monkeypatch.setattr(answers, "chat_model", lambda key: MockLLM())
    monkeypatch.setattr(CitationQueryEngine, "asynthesize", never)
    body = ask({"question": "What is the capital of Australia?", "stream": False}).json()
    assert body["refused"] and body["answer"].startswith(answers.A.NOT_COVERED) and body["sources"] == []


def test_rag_embeds_on_the_callers_gemini_key_when_given_and_the_owners_otherwise(offline_rag):
    ask({"question": QUESTION, "stream": False}, BOTH)
    assert EMBEDDED_FOR == ["AIza-caller"]
    ask({"question": QUESTION, "stream": False}, GROQ, address="203.0.113.2")
    assert EMBEDDED_FOR == ["AIza-caller"]      # none made: the index's own, on the owner's key


def test_async_search_finds_what_the_sync_search_finds_with_one_embedding():
    """LlamaIndex's async vector search drops the question's embedding, and the fused search made a second one --
    two Gemini embeddings a question, counted. CookbookRetriever._aretrieve makes it once and keeps it."""
    from llama_index.core import MockEmbedding, QueryBundle

    made = []

    class Counting(MockEmbedding):
        async def _aget_query_embedding(self, query):
            made.append(query)
            return await super()._aget_query_embedding(query)

    embed = Counting(embed_dim=answers.A.EMBED_DIM)
    engine = answers.A.engine(answers.INDEX, MockLLM(), floor=-1, embed=embed)
    found = asyncio.run(engine.aretrieve(QueryBundle("How does the runtime detect a doom loop?")))
    expected = engine.retrieve(QueryBundle("How does the runtime detect a doom loop?"))
    assert [n.node.node_id for n in found] == [n.node.node_id for n in expected]
    assert len(made) == 1


@pytest.mark.parametrize("question", ["How does the runtime detect a doom loop?", "double_send LangGraph",
                                      "Why is CrewAI pinned to an old version?", "memory stores and recall ranking"])
def test_sharing_bm25_and_searching_once_find_the_same_passages_in_the_same_order(question):
    """BM25 is built once and shared, and the fusion is handed the vector search's results instead of searching
    again (both measured as most of a question's CPU). The passages must be exactly what the old way found."""
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    index = answers.INDEX
    old = QueryFusionRetriever([index.as_retriever(similarity_top_k=10),
                                BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)],
                               llm=MockLLM(), num_queries=1, mode="reciprocal_rerank", similarity_top_k=5,
                               use_async=False)
    new = answers.A.CookbookRetriever(index, MockLLM(), floor=None)
    expected = [(n.node.node_id, round(n.score, 6)) for n in old.retrieve(QueryBundle(question))]
    assert [(n.node.node_id, round(n.score, 6)) for n in new.retrieve(QueryBundle(question))] == expected
    assert [(n.node.node_id, round(n.score, 6)) for n in asyncio.run(new.aretrieve(QueryBundle(question)))] == expected
    assert new.bm25 is answers.A.CookbookRetriever(index, MockLLM()).bm25          # one BM25, shared


@pytest.mark.parametrize("question", ["How does the runtime detect a doom loop?", "double_send LangGraph",
                                      "Why is CrewAI pinned to an old version?", "memory stores and recall ranking",
                                      "What is the capital of Australia?"])
def test_the_kept_matrix_finds_what_llamaindex_finds_with_the_same_scores(question):
    """The vector search on a matrix kept between questions (search.nearest) against LlamaIndex's own: the same
    passages, in the same order, and the same raw cosine -- the refusal floor is judged on it. A query vector made
    from the question's words, so each question is near different passages."""
    import random

    from llama_index.core import MockEmbedding, QueryBundle

    class Varied(MockEmbedding):
        def _get_query_embedding(self, query):
            rng = random.Random(query)
            return [rng.uniform(-1, 1) for _ in range(answers.A.EMBED_DIM)]

    index = answers.INDEX
    theirs = index.as_retriever(similarity_top_k=10, embed_model=Varied(embed_dim=answers.A.EMBED_DIM))
    expected = [(n.node.node_id, round(n.score, 9)) for n in theirs.retrieve(QueryBundle(question))]
    query = QueryBundle(question, embedding=Varied(embed_dim=answers.A.EMBED_DIM).get_query_embedding(question))
    assert [(n.node.node_id, round(n.score, 9)) for n in answers.A.nearest(theirs, query, 10)] == expected


# --------------------------------------------------------------------------- agentic RAG

class Searcher:
    """A client that keeps searching, a new query each call, `delay` seconds a call -- a run long enough to leave."""

    QUERIES = ["memory stores and recall ranking", "doom loop detection in the runtime",
               "context selection within a token budget", "LangGraph interrupt and resume", "tool registry schemas",
               "compaction of older turns", "supervisor delegation to workers", "MCP transport and sessions"]

    def __init__(self, delay=0.0):
        self.sent, self.delay, self.chat = [], delay, self
        self.completions = self

    def create(self, **kwargs):
        time.sleep(self.delay)
        self.sent.append(kwargs)
        n = len(self.sent)
        if "tools" in kwargs:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"c{n}", "type": "function",
                "function": {"name": "search_cookbook", "arguments": json.dumps({"query": self.QUERIES[n - 1]})}}]}
        else:
            message = {"role": "assistant", "content": "What the searches found [02_agent_runtime/runtime.py]."}
        return ChatCompletion.model_validate({
            "id": "f", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})


def test_agentic_runs_on_the_callers_keys_and_streams_its_steps(fake_client, monkeypatch):
    made = []
    monkeypatch.setattr(answers, "client", lambda key: made.append(fake_client(key)) or made[-1])
    events = events_of(ask({"question": QUESTION, "mode": "agentic"}, BOTH).text)
    answer = dict(events)["answer"]
    assert "Answered on gsk_visitor" in answer["answer"]
    assert f"/blob/{answers.BUILT['commit']}/02_agent_runtime/runtime.py" in answer["answer"]
    assert [c["tool"] for c in answer["tool_calls"]] == ["search_cookbook"]
    # the step it is on, as the page shows it (this stand-in run is over before a second look at it)
    assert any(name == "progress" and "of up to 8" in data["message"] for name, data in events)
    assert EMBEDDED_FOR == ["AIza-caller"]      # its searches embed on the caller's Gemini key


# --------------------------------------------------------------------------- limits, and many at once

def test_over_the_per_minute_limit_one_address_waits_and_others_do_not(offline_rag):
    for _ in range(api.PER_MINUTE):
        assert ask({"question": QUESTION, "stream": False}).status_code == 200
    over = ask({"question": QUESTION, "stream": False})
    assert over.status_code == 429 and int(over.headers["Retry-After"]) > 0
    assert ask({"question": QUESTION, "stream": False}, address="198.51.100.7").status_code == 200


def test_the_minute_is_a_sliding_window():
    now = [0.0]
    limit = api.PerMinute(2, clock=lambda: now[0])
    assert limit.wait("a") == 0 and limit.wait("a") == 0
    assert limit.wait("a") == pytest.approx(60)
    now[0] = 30
    assert limit.wait("a") == pytest.approx(30) and limit.wait("b") == 0
    now[0] = 60
    assert limit.wait("a") == 0                 # the first request has left the window


def slow_rag(seconds, running, most):
    """An async RAG that takes `seconds`, awaiting as the real one awaits Gemini and Groq; it records how many run at
    once."""
    async def rag(question, key, stage, embed=None):
        stage["now"] = "Searching the cookbook"
        running.append(1)
        most.append(len(running))
        try:
            await asyncio.sleep(seconds)
        finally:
            running.pop()
        return {"refused": True, "answer": answers.A.NOT_COVERED, "response": None, "tokens": 0}
    return rag


def test_many_requests_are_worked_out_at_once_on_one_thread(monkeypatch):
    """Eight half-second answers, sent together, all done in about half a second -- not four -- with no thread per
    request: while one waits on the network, the event loop serves the others."""
    running, most = [], []
    monkeypatch.setattr(answers, "arag", slow_rag(0.5, running, most))
    started = time.monotonic()
    replies = send(*[("POST", "/api/ask", {"json": {"question": QUESTION, "stream": False},
                                           "headers": {**GROQ, "X-Forwarded-For": f"203.0.113.{n}"}})
                     for n in range(8)])
    took = time.monotonic() - started
    assert all(r.status_code == 200 for r in replies)
    assert max(most) == 8 and took < 1.5


def test_over_the_runs_at_once_a_request_waits_its_turn(monkeypatch):
    monkeypatch.setattr(api, "GATES", api.Gates({"rag": 2, "agentic": 1}))
    running, most = [], []
    monkeypatch.setattr(answers, "arag", slow_rag(0.3, running, most))
    replies = send(*[("POST", "/api/ask", {"json": {"question": QUESTION},
                                           "headers": {**GROQ, "X-Forwarded-For": f"203.0.113.{n}"}})
                     for n in range(5)])
    assert max(most) == 2                                           # never more than two at once
    assert all(dict(events_of(r.text)).get("answer") for r in replies)     # and every one answered
    waited = [r for r in replies if any(d.get("message", "").startswith("Waiting for a free slot")
                                        for _, d in events_of(r.text))]
    assert len(waited) >= 3


# --------------------------------------------------------------------------- a caller who leaves

def test_a_caller_who_leaves_stops_rag_at_once(monkeypatch):
    seen = []

    async def rag(question, key, stage, embed=None):
        stage["now"] = "Searching the cookbook"
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            seen.append("cancelled")
            raise
    monkeypatch.setattr(answers, "arag", rag)

    async def scenario():
        stream = api.events(api.Ask(question=QUESTION), "gsk_visitor", None)
        assert "Searching the cookbook" in await stream.__anext__()
        await stream.aclose()                   # what the server does when the connection drops
        await asyncio.sleep(0.05)
    asyncio.run(scenario())
    assert seen == ["cancelled"]


def test_a_caller_who_leaves_stops_an_agent_run_before_its_next_model_call(monkeypatch):
    """The run is on a thread, which cannot be killed: it is stopped where its next model call would go out. Its
    slot comes back only once the thread has ended."""
    client = Searcher(delay=0.2)
    monkeypatch.setattr(answers, "client", lambda key: client)

    async def scenario():
        stream = api.events(api.Ask(question=QUESTION, mode="agentic"), "gsk_visitor", "AIza-caller")
        await stream.__anext__()
        while len(client.sent) < 2:
            await asyncio.sleep(0.05)
        await stream.aclose()
        left_at = len(client.sent)
        gate = api.GATES("agentic")
        for _ in range(100):                    # its slot: back when the thread has ended
            if gate._value == api.RUNS_AT_ONCE["agentic"]:
                break
            await asyncio.sleep(0.05)
        return left_at, gate._value
    left_at, free = asyncio.run(scenario())
    assert free == api.RUNS_AT_ONCE["agentic"]
    assert len(client.sent) <= left_at + 1      # at most the call already in flight
    assert len(client.sent) < 8                 # far short of a whole run


# --------------------------------------------------------------------------- errors, in words

def error(kind, status, message="refused"):
    return kind(message, response=httpx.Response(status, request=httpx.Request("POST", "https://x")), body=None)


@pytest.mark.parametrize("failure, status, says", [
    (error(openai.AuthenticationError, 401), 401, "That Groq key was refused"),
    (error(openai.RateLimitError, 429, "Rate limit reached on tokens per day (TPD)"), 429, "daily limit"),
    (error(openai.RateLimitError, 429, "Rate limit reached on tokens per minute (TPM)"), 429, "per-minute limit"),
    (RuntimeError("something broke"), 500, "Something went wrong"),
])
def test_errors_reach_the_caller_as_words_and_never_the_key(fake_client, monkeypatch, capsys, failure, status, says):
    monkeypatch.setattr(answers, "client", lambda key: fake_client(key, fail=failure))
    events = events_of(ask({"question": QUESTION, "mode": "agentic"}, BOTH).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == status and says in events[-1][1]["message"]
    printed = capsys.readouterr()
    assert "gsk_visitor" not in printed.out + printed.err and "AIza-caller" not in printed.out + printed.err


def test_a_key_that_passed_its_check_is_remembered_as_a_hash_and_a_refused_one_is_not(monkeypatch):
    calls = []

    class Groq:
        def __init__(self, api_key, **kwargs):
            self.key, self.models = api_key, self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list(self):
            calls.append(self.key)
            if "bad" in self.key:
                raise error(openai.AuthenticationError, 401)

    monkeypatch.undo()                          # the real check_key, not the fixture's stand-in
    monkeypatch.setattr(api.openai, "AsyncOpenAI", Groq)
    monkeypatch.setattr(api, "_ACCEPTED", {})
    assert asyncio.run(api.check_key("groq", "gsk_good")) and asyncio.run(api.check_key("groq", "gsk_good"))
    assert calls == ["gsk_good"]                # checked once
    assert "gsk_good" not in str(api._ACCEPTED)
    assert not asyncio.run(api.check_key("groq", "gsk_bad")) and not asyncio.run(api.check_key("groq", "gsk_bad"))
    assert calls == ["gsk_good", "gsk_bad", "gsk_bad"]          # a refused key is asked about again


# --------------------------------------------------------------------------- one server, page and API

def test_the_page_and_the_api_are_served_together(app):
    app.demo.launch(prevent_thread_lock=True, quiet=True, **app.LAUNCH)
    try:
        url = app.demo.local_url.rstrip("/")
        health = httpx.get(f"{url}/api/health").json()
        assert health["status"] == "ok" and health["index"]["commit"] == answers.BUILT["commit"]
        assert "gradio" in httpx.get(f"{url}/").text.lower()
        paths = httpx.get(f"{url}/api/openapi.json").json()["paths"]
        assert set(paths) == {"/api/ask", "/api/health"}        # the API's docs: its routes, none of Gradio's
        assert httpx.get(f"{url}/api/docs").status_code == 200
    finally:
        app.demo.close()
