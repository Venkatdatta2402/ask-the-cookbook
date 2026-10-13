"""The chat: follow-ups, the meter, Compact now, and what each costs the demo key's day."""

import json

import pytest
from openai.types.chat import ChatCompletion

from ask_the_cookbook.runtime import llm as shared
from ask_the_cookbook.runtime.selection import select
from ask_the_cookbook.session import KEEP_RECENT, Session


def completion(content, model="m"):
    return ChatCompletion.model_validate({
        "id": "f", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


class Compactor:
    """A client for compaction: a summary when asked to summarise, a list of facts when asked for JSON."""

    def __init__(self):
        self.calls, self.chat = [], self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            return completion(json.dumps({"facts": [{"kind": "fact", "text": "Chapter 4 ranks memories by recency."}]}))
        return completion("The visitor asked about memory ranking and doom loops.")


def chat_of(exchanges):
    session = Session()
    for n in range(exchanges):
        session.add(f"Question {n} about chapter {n} and its memory store, at some length to have weight?",
                    f"Answer {n}: chapter {n} explains it, with a few sentences of detail about how it works. " * 3)
    return session


@pytest.fixture
def no_shared_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(shared.client.chat.completions, "create", lambda **kw: calls.append(kw))
    return calls


# --------------------------------------------------------------------------- holding, and follow-ups

def test_the_meter_counts_what_the_chat_holds():
    session = Session()
    assert session.held() == 0
    session.add("What is a doom loop?", "The same call again.")
    assert 0 < session.held() < 30


@pytest.fixture
def rag_offline(app, monkeypatch):
    """RAG on a stand-in LLM with no floor, recording what retrieval was asked."""
    from llama_index.core.llms import MockLLM
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    monkeypatch.setattr(app.A, "FLOOR", -1)
    searched, real_ask = [], app.A.ask
    monkeypatch.setattr(app.A, "ask", lambda engine, q: searched.append(q) or real_ask(engine, q))
    return searched


def test_a_rag_follow_up_is_answered_as_asked(app, ask, rag_offline):
    session = Session()
    ask("How does chapter 4 rank memories?", session=session)
    ask("and chapter 5?", session=session)
    assert rag_offline == ["How does chapter 4 rank memories?", "and chapter 5?"]     # no rewrite, no extra call


def test_ask_about_puts_the_passage_into_the_question(app, ask, rag_offline):
    session = Session()
    passage = "after fusion a score is rank arithmetic"
    ask("Why does that matter for the floor?", session=session, quote=passage)
    assert passage in rag_offline[0] and "Why does that matter for the floor?" in rag_offline[0]
    assert session.shown[0]["content"] == f"> {passage}\n\nWhy does that matter for the floor?"
    assert passage in session.history[0]["content"]          # later turns know what "that" was


def test_the_quote_box_is_emptied_once_sent_and_kept_when_turned_away(app):
    session = Session()
    *_, question_box, quote_box = list(app.respond("", "a passage", "", app.RAG, session))[-1]
    assert quote_box == "a passage"                          # nothing sent: the selection stays
    sending = app.respond("q", "a passage", "gsk_x", app.AGENTIC, session)
    *_, question_box, quote_box = next(sending)              # the first update: the message is on its way
    sending.close()                                          # stop there: nothing is sent to a model
    assert (question_box, quote_box) == ("", "")


def test_a_selection_over_the_limit_is_refused_before_any_call(app, ask):
    from ask_the_cookbook.session import QUOTE_LIMIT
    reply, _ = ask("Why?", quote="x" * (QUOTE_LIMIT + 1))
    assert f"at most {QUOTE_LIMIT:,} characters" in reply
    assert app.DAILY.left() == app.DAILY_LIMIT


def test_agentic_rag_has_the_passage_pinned_in_its_context(app, ask, fake_client, monkeypatch):
    made = []
    monkeypatch.setattr(app, "client", lambda key: made.append(fake_client(key, search=False)) or made[-1])
    passage = "the floor is applied on raw vector scores, before fusion"
    ask("Why before?", key="gsk_visitor", mode=app.AGENTIC, quote=passage)
    request = [m for m in made[0].sent[0]["messages"] if m.get("role") == "user"][-1]["content"]
    assert passage in request and "Why before?" in request


def test_a_follow_up_turned_away_points_to_ask_about(app, ask, monkeypatch):
    from llama_index.core.llms import MockLLM
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    first, _ = ask("What is the capital of Australia?")              # the floor turns it away: no chat yet
    assert "Ask about" not in first
    session = Session()
    session.add("How does chapter 4 rank memories?", "By relevance and recency.")
    later, _ = ask("and the other one?", session=session)
    assert "Ask about" in later
    assert "Ask about" not in session.history[-1]["content"]         # the hint is for the visitor, not the model


def test_the_page_carries_the_ask_about_parts(app):
    import httpx
    app.demo.launch(prevent_thread_lock=True, quiet=True, js=app.ASK_ABOUT_JS, css=app.ASK_ABOUT_CSS)
    try:
        assert httpx.get(app.demo.local_url).status_code == 200
        config = httpx.get(app.demo.local_url.rstrip("/") + "/config").text
        assert "ask-about-trigger" in config and "ask-about-chip" in config
    finally:
        app.demo.close()


def test_numbered_citations_are_kept_as_file_and_lines(app, monkeypatch):
    """A passage as CitationQueryEngine hands it over ("Source 1:" and then the text), placed in its file."""
    from types import SimpleNamespace
    from llama_index.core.schema import TextNode
    from ask_the_cookbook import corpus
    lines = (corpus.REPO / "02_agent_runtime" / "runtime.py").read_text(encoding="utf-8").splitlines()
    start = next(n for n, line in enumerate(lines, 1) if line.startswith("def _similar_args"))
    passage = "\n".join(lines[start - 1:start + 4])
    node = TextNode(text=f"Source 1:\n{passage}\n", metadata={"file": "02_agent_runtime/runtime.py"})
    response = SimpleNamespace(source_nodes=[SimpleNamespace(node=node)])
    monkeypatch.setattr(app, "chat_model", lambda key: None)
    monkeypatch.setattr(app.A, "engine", lambda index, llm: None)
    monkeypatch.setattr(app.A, "ask", lambda engine, q: {
        "refused": False, "answer": "Compared on values [1].", "response": response})
    shown, kept = app.ask_engine("How are calls compared?", "gsk_x", Session())
    assert kept == f"Compared on values [02_agent_runtime/runtime.py, lines {start}-{start + 4}]."
    assert "Compared on values [1]." in shown and "**Sources**" in shown


def test_every_message_on_the_demo_key_counts(app, ask, monkeypatch):
    from llama_index.core.llms import MockLLM
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    session = Session()
    ask("What is a doom loop?", session=session)
    ask("And how is it detected?", session=session)
    assert app.DAILY.left() == app.DAILY_LIMIT - 2


def test_the_agent_starts_from_the_chat(app, ask, fake_client, monkeypatch):
    made = []
    monkeypatch.setattr(app, "client", lambda key: made.append(fake_client(key, search=False)) or made[-1])
    session = Session()
    session.add("How does chapter 4 rank memories?", "By relevance and recency.")
    session.notes = ["Earlier: the visitor asked about doom loops."]
    reply, _ = ask("And how does chapter 5 use that?", key="gsk_visitor", mode=app.AGENTIC, session=session)
    sent = " ".join(str(m.get("content")) for m in made[0].sent[0]["messages"])
    assert "How does chapter 4 rank memories?" in sent
    assert "Earlier: the visitor asked about doom loops." in sent
    assert "Answered on gsk_visitor" in reply


# --------------------------------------------------------------------------- Compact now

def test_compaction_keeps_the_recent_turns_and_summarises_the_rest(no_shared_calls):
    session, client = chat_of(6), Compactor()
    recent = session.history[-KEEP_RECENT:]
    assert session.can_compact()
    before, after = session.compact(client)
    assert session.history == recent
    assert session.notes[0] == "The visitor asked about memory ranking and doom loops."
    assert "Chapter 4 ranks memories by recency." in session.notes
    assert after < before == chat_of(6).held()
    assert len(client.calls) == 2 and no_shared_calls == []


def test_a_short_chat_has_nothing_to_compact(app):
    session = chat_of(KEEP_RECENT // 2)
    assert not session.can_compact()
    *_, status = list(app.compact_now("", session))[-1]
    assert "Nothing to compact yet" in status
    assert app.DAILY.left() == app.DAILY_LIMIT              # nothing spent, nothing counted


def test_compact_now_on_a_visitors_key_uses_their_key(app, monkeypatch, no_shared_calls):
    visitor = Compactor()
    monkeypatch.setattr(app, "client", lambda key: visitor)
    session = chat_of(6)
    shown, session, status = list(app.compact_now("gsk_visitor", session))[-1]
    assert "Older turns compacted" in shown[-1]["content"]
    assert visitor.calls and no_shared_calls == []
    assert app.DAILY.left() == app.DAILY_LIMIT
    assert session.notes and f"Chat holds **{session.held():,} tokens**" in status


def test_compact_now_on_the_demo_key_counts_as_a_question(app, monkeypatch):
    owner = Compactor()
    monkeypatch.setattr(shared.client.chat.completions, "create", owner.create)
    list(app.compact_now("", chat_of(6)))
    assert owner.calls
    assert app.DAILY.left() == app.DAILY_LIMIT - 1


def test_after_compaction_the_agent_gets_the_summary(app, ask, fake_client, monkeypatch, no_shared_calls):
    made = []
    monkeypatch.setattr(app, "client", lambda key: Compactor() if not made else made[-1])
    session = chat_of(6)
    list(app.compact_now("gsk_visitor", session))
    made.append(fake_client("gsk_visitor", search=False))
    ask("Which chapter was that?", key="gsk_visitor", mode=app.AGENTIC, session=session)
    sent = " ".join(str(m.get("content")) for m in made[0].sent[0]["messages"])
    assert "The visitor asked about memory ranking and doom loops." in sent


def test_a_compactions_notes_count_as_the_oldest_items():
    """At the start of an agent run, the two items selection always keeps (keep_last=2) are the question and the
    previous answer -- not the notes, which stand for the chat's oldest turns."""
    from ask_the_cookbook import agent
    from ask_the_cookbook.runtime.chat import Conversation
    from ask_the_cookbook.runtime.context import available, tokens

    conversation = Conversation(system="Answer about the cookbook.")
    conversation.messages += [{"role": "user", "content": "How does chapter 4 rank memories?"},
                              {"role": "assistant", "content": "By relevance and recency."},
                              {"role": "user", "content": "And chapter 5?"}]
    conversation.notes = ["Summary: the visitor asked about doom loops earlier.", "Chapter 2 detects doom loops."]
    items = available(conversation)
    assert max(i.turn for i in items if i.source == "notes") < min(i.turn for i in items if i.kind == "message")

    must = [i for i in items if i.pinned] + [i for i in items if i.content == "assistant: By relevance and recency."]
    kept = select(items, budget=tokens(must), keep_last=agent.KEEP_LAST).kept    # room for the must-keeps only
    assert "assistant: By relevance and recency." in [i.content for i in kept]
    assert not [i for i in kept if i.source == "notes"]


def test_new_chat_starts_empty(app):
    shown, session, status = app.new_chat()
    assert shown == [] and session.history == [] and session.notes == []
    assert "Chat holds **0 tokens**. The agent can take up to **5,000** per model call." in status
