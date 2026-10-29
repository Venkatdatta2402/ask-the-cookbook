"""Agent runs at once on different keys stay apart: each run's calls go out on its own client, and are counted
against its own budget. The shared client (chapter 1's, made with the owner's key) is never used by them."""

import concurrent.futures

from ask_the_cookbook import agent
from ask_the_cookbook import answers
from ask_the_cookbook import search
from ask_the_cookbook.runtime import llm


def test_three_runs_at_once_stay_apart(app, ask, fake_client, monkeypatch):
    def shared(*args, **kwargs):
        raise AssertionError("an agent run on a visitor's key called the shared client")
    monkeypatch.setattr(llm.client.chat.completions, "create", shared)

    clients = {}

    def make(key):
        clients[key] = fake_client(key, prompt_tokens=1000 if key.endswith("two") else 100)
        return clients[key]
    monkeypatch.setattr(answers, "client", make)

    runs = {f"gsk_visitor_{n}": f"A question from visitor {n} about memory" for n in ("one", "two", "three")}
    with concurrent.futures.ThreadPoolExecutor(len(runs)) as pool:
        replies = dict(zip(runs, pool.map(lambda key: ask(runs[key], key, app.AGENTIC)[0], runs)))

    for key, question in runs.items():
        sent = clients[key].sent
        assert len(sent) == 2
        assert all(any(m.get("content") == question for m in kw["messages"]) for kw in sent)
        assert f"Answered on {key}" in replies[key]
        assert f"{2 * (1000 if key.endswith('two') else 100) + 20:,} tokens on your key" in replies[key]


def test_repairing_a_failed_tool_call_uses_the_runs_key(app, ask, monkeypatch):
    """A tool that raises (a search when Gemini's quota is spent) is retried with arguments chapter 2's DebugThink
    asks the model for. That request is a model call like any other, and goes on the visitor's key."""
    from openai.types.chat import ChatCompletion

    shared_calls = []
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **kw: shared_calls.append(kw))

    def quota_spent(self, query_bundle):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(search.CookbookRetriever, "_retrieve", quota_spent)

    class Visitor:
        def __init__(self):
            self.repairs, self.chat = 0, self
            self.completions = self

        def create(self, **kwargs):
            if "response_format" in kwargs:                   # DebugThink asking for fixed arguments
                self.repairs += 1
                message = {"role": "assistant", "content": '{"query": "recall ranking"}'}
            elif not any(m.get("role") == "tool" for m in kwargs["messages"]):
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "search_cookbook", "arguments": '{"query": "recall"}'}}]}
            else:
                message = {"role": "assistant", "content": "The search failed, so I cannot say."}
            return ChatCompletion.model_validate({
                "id": "f", "object": "chat.completion", "created": 0, "model": kwargs["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})

    visitor = Visitor()
    monkeypatch.setattr(answers, "client", lambda key: visitor)
    reply = ask("How does recall rank?", "gsk_visitor", app.AGENTIC)[0]
    assert "The search failed" in reply
    assert visitor.repairs == 2             # three attempts: two repairs between them
    assert shared_calls == []


def test_without_a_client_a_run_uses_the_shared_one_and_puts_it_back(index, fake_client, monkeypatch):
    shared = fake_client("shared", search=False)
    monkeypatch.setattr(llm.client.chat.completions, "create", shared.create)
    before = llm.client.chat.completions.create
    budget = search.Budget(label="test")
    result = agent.run(index, "A question on the shared client", budget)
    assert result["answer"].startswith("Answered on shared")
    assert budget.calls == 1
    assert llm.client.chat.completions.create == before
