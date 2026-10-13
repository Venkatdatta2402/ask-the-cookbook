"""The page's answers: what is checked before a model is called, the daily limit, the engine's sources, the agent,
and errors in words. A stand-in LLM answers the engine; a fake client answers the agent."""

import os

import httpx
import openai
import pytest
from llama_index.core.llms import MockLLM


@pytest.fixture
def engine_offline(app, monkeypatch):
    """The engine on a stand-in LLM, and no floor: a stand-in embedding is near nothing, so with the floor every
    question would be turned away before the sources could be seen."""
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    monkeypatch.setattr(app.A, "FLOOR", -1)


@pytest.mark.parametrize("question, key, mode, says", [
    ("", "", "quick", "Type a question first"),
    ("x" * 501, "", "quick", "under 500 characters"),
    ("q", "sk-not-groq", "quick", "isn't a Groq API key"),
    ("q", "hf_a_hugging_face_token", "agent", "isn't a Groq API key"),
    ("q", "", "agent", "runs on your own key"),
])
def test_turned_away_before_any_model_call(app, ask, question, key, mode, says):
    reply, _ = ask(question, key, app.AGENTIC if mode == "agent" else app.RAG)
    assert says in reply
    assert app.DAILY.left() == app.DAILY_LIMIT


def test_the_engine_answers_with_linked_sources(app, ask, engine_offline):
    reply, status = ask("What counts as a doom loop?")
    assert "**Sources**" in reply
    assert f"/blob/{app.BUILT['commit']}/" in reply
    assert f"**{app.DAILY_LIMIT - 1} of {app.DAILY_LIMIT}**" in status


def test_with_the_floor_a_question_nothing_is_near_is_turned_away(app, ask, monkeypatch):
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    reply, _ = ask("What is the capital of Australia?")
    assert reply.startswith(app.A.NOT_COVERED)


def test_the_daily_limit_stops_the_demo_key_but_not_a_visitors(app, ask, engine_offline):
    for _ in range(app.DAILY_LIMIT):
        assert app.DAILY.take()
    assert "free questions are used up" in ask("What counts as a doom loop?")[0]
    assert "**Sources**" in ask("What counts as a doom loop?", key="gsk_visitor")[0]


def test_the_count_starts_again_on_a_new_day(app):
    for _ in range(app.DAILY_LIMIT):
        app.DAILY.take()
    assert not app.DAILY.take()
    app.DAILY.day = app.DAILY.day.replace(year=app.DAILY.day.year - 1)
    assert app.DAILY.take()
    assert app.DAILY.left() == app.DAILY_LIMIT - 1


def test_the_engine_key_swap_puts_the_owners_key_back(app):
    llm = app.chat_model("gsk_visitor")
    assert llm._client.api_key == "gsk_visitor"
    assert os.environ["GROQ_API_KEY"] == "gsk_placeholder_owner"


def test_the_agent_answers_on_the_visitors_key_with_linked_citations(app, ask, fake_client, monkeypatch):
    made = []
    monkeypatch.setattr(app, "client", lambda key: made.append(fake_client(key)) or made[-1])
    reply, _ = ask("What counts as a doom loop?", key="gsk_visitor", mode=app.AGENTIC)
    assert "Answered on gsk_visitor" in reply
    assert f"/blob/{app.BUILT['commit']}/02_agent_runtime/runtime.py" in reply
    assert "1 tool calls" in reply
    assert [kw["model"] for kw in made[0].sent] == [app.A.MODEL] * 2


@pytest.mark.parametrize("status, kind, says", [
    (401, openai.AuthenticationError, "That key was refused"),
    (429, openai.RateLimitError, "hit its rate limit"),
])
def test_errors_reach_the_visitor_as_words(app, ask, fake_client, monkeypatch, status, kind, says):
    error = kind("refused", response=httpx.Response(status, request=httpx.Request("POST", "https://x")), body=None)
    monkeypatch.setattr(app, "client", lambda key: fake_client(key, fail=error))
    assert says in ask("q", key="gsk_visitor", mode=app.AGENTIC)[0]


def test_the_page_loads(app):
    app.demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        assert httpx.get(app.demo.local_url).status_code == 200
    finally:
        app.demo.close()
