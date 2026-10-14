"""The chat: follow-ups, the meter, Compact now, and what each costs the demo key's day."""

import json

import pytest
from openai.types.chat import ChatCompletion

from ask_the_cookbook.runtime import llm as shared
from ask_the_cookbook.runtime.selection import select
from ask_the_cookbook.session import COMPACT_WORTH, KEEP_RECENT, Session


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
    """A chat whose answers are the size of real ones (~400 tokens): three old exchanges are worth compacting."""
    session = Session()
    for n in range(exchanges):
        session.add(f"Question {n} about chapter {n} and its memory store, at some length to have weight?",
                    f"Answer {n}: chapter {n} explains it, with a few sentences of detail about how it works. " * 20)
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
    searched, real_rag = [], app.rag
    monkeypatch.setattr(app, "rag", lambda q, key, stage: searched.append(q) or real_rag(q, key, stage))
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
    import gradio as gr
    import httpx
    app.demo.launch(prevent_thread_lock=True, quiet=True, js=app.ASK_ABOUT_JS, css=app.CSS, theme=gr.themes.Soft())
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
    monkeypatch.setattr(app, "rag", lambda q, key, stage: {
        "refused": False, "answer": "Compared on values [1].", "response": response})
    shown, kept = app.ask_engine("How are calls compared?", "gsk_x", Session())
    assert kept == f"Compared on values [02_agent_runtime/runtime.py, lines {start}-{start + 4}]."
    at = f"{app.blob('02_agent_runtime/runtime.py')}#L{start}-L{start + 4}"
    assert f"Compared on values [[1]]({at})." in shown                       # the citation opens the lines
    assert f"1. [02_agent_runtime/runtime.py, lines {start}-{start + 4}]({at})" in shown


def passage(file, text):
    from llama_index.core.schema import TextNode
    return TextNode(text=f"Source 1:\n{text}", metadata={"file": file})


def test_a_readme_source_opens_at_its_lines_named_by_its_heading(app):
    from ask_the_cookbook import corpus
    file = "04_memory/README.md"
    lines = (corpus.REPO / file).read_text(encoding="utf-8").splitlines()
    heading = next(n for n, line in enumerate(lines) if line.startswith("## "))
    start = next(n for n, line in enumerate(lines, 1) if n > heading + 1 and len(line.strip()) >= 30
                 and not line.startswith("#"))
    name, url = app.source(passage(file, "\n".join(lines[start - 1:start + 2])))
    assert name == f"{file}, lines {start}-{start + 2} § {lines[heading][3:].strip()}"
    assert url == f"{app.blob(file)}?plain=1#L{start}-L{start + 2}"


def test_a_module_source_opens_at_its_lines(app):
    from ask_the_cookbook import corpus
    file = "04_memory/stores.py"
    lines = (corpus.REPO / file).read_text(encoding="utf-8").splitlines()
    start = next(n for n, line in enumerate(lines, 1) if len(line.strip()) >= 30)
    assert app.source(passage(file, "\n".join(lines[start - 1:start + 2]))) == (
        f"{file}, lines {start}-{start + 2}", f"{app.blob(file)}#L{start}-L{start + 2}")


def test_a_notebook_source_opens_on_github_named_by_its_heading(app):
    from ask_the_cookbook import corpus
    file = "02_agent_runtime/notebooks/02_the_agent_loop.ipynb"
    prose = corpus._notebook_prose(corpus.REPO / file).splitlines()
    at = prose.index("## `act.py`: executing a decision already made")
    text = next(line for line in prose[at + 1:] if len(line.strip()) >= 30)
    node = passage(file, text)
    assert app.source(node) == (f"{file} § `act.py`: executing a decision already made", app.blob(file))
    assert app.where(node).startswith(f"{file}, lines ")          # the chat still keeps its lines, for the agent


def test_a_comment_in_a_code_example_is_not_a_heading(app, monkeypatch):
    text = "## Setup\nRun this:\n```bash\n# install the packages\npip install x\n```\nThen go on."
    monkeypatch.setattr(app.A, "file_text", lambda file: text)
    assert app.heading_above("any.md", 7) == "Setup"


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
    _, _, status, _ = list(app.compact_now("", session))[-1]
    assert "Not worth compacting yet: 0 tokens" in status
    assert app.DAILY.left() == app.DAILY_LIMIT              # nothing spent, nothing counted


def test_small_old_turns_are_not_worth_compacting(app, no_shared_calls):
    """The live case: old turns too small for a summary and facts to come out any smaller."""
    session = Session()
    for n in range(5):
        session.add(f"Short question {n}?", f"Short answer {n}.")
    assert 0 < session.compactable() < COMPACT_WORTH and not session.can_compact()
    _, _, status, _ = list(app.compact_now("", session))[-1]
    assert "Not worth compacting yet" in status and no_shared_calls == []
    assert app.DAILY.left() == app.DAILY_LIMIT


def test_a_compaction_that_comes_out_larger_is_undone(app, monkeypatch):
    class Verbose(Compactor):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "response_format" in kwargs:
                facts = [{"kind": "fact", "text": f"A long fact number {n}, restating the chat at length. " * 10}
                         for n in range(30)]
                return completion(json.dumps({"facts": facts}))
            return completion("A summary far longer than what it replaces. " * 200)

    monkeypatch.setattr(app, "client", lambda key: Verbose())
    session = chat_of(6)
    history, held = list(session.history), session.held()
    shown, session, status, _ = list(app.compact_now("gsk_visitor", session))[-1]
    assert session.history == history and session.notes == [] and session.held() == held
    assert "would not have made the chat any smaller" in status
    assert not any("compacted" in m["content"] for m in shown)


@pytest.mark.parametrize("answer, kept", [
    ("Ranked [1].", "Ranked [a.py, lines 1-5]."),
    ("Ranked [1][2].", "Ranked [a.py, lines 1-5][b.md, lines 7-9]."),
    ("Ranked [1, 2].", "Ranked [a.py, lines 1-5; b.md, lines 7-9]."),
    ("Ranked [Source 2].", "Ranked [b.md, lines 7-9]."),
    ("Ranked [Source 1, commentary in the same file].", "Ranked [a.py, lines 1-5, commentary in the same file]."),
    ("Exact facts【1†L9-L15】【2†L13-L20】.", "Exact facts[a.py, lines 1-5][b.md, lines 7-9]."),
    ("A list [3] and [0.5] stay.", "A list [3] and [0.5] stay."),
])
def test_every_citation_form_gpt_oss_writes_is_kept_as_file_and_lines(app, answer, kept):
    assert app.cited(answer, {1: "a.py, lines 1-5", 2: "b.md, lines 7-9"}) == kept


def test_code_is_never_read_as_citations(app):
    """Live, inside a table: `pairs[0] == pairs[2] and pairs[1] == pairs[3]` had [2], [1], [3] linked to sources."""
    code = "`pairs[0] == pairs[2] and pairs[1] == pairs[3] and pairs[0] != pairs[1]`"
    links = {1: "u1", 2: "u2", 3: "u3"}
    assert app.linked(f"| Oscillation | {code} |", links) == f"| Oscillation | {code} |"
    fenced = "```python\nif seen[1] == seen[2]:\n    stop()\n```"
    assert app.linked(f"Like this [1]:\n{fenced}", links) == f"Like this [[1]](u1):\n{fenced}"
    assert app.linked("an index pairs[2] outside code", links) == "an index pairs[2] outside code"
    assert app.cited(f"See {code} [2].", {2: "a.py, lines 1-5"}) == f"See {code} [a.py, lines 1-5]."
    assert app.link_files("`[04_memory/recall.py]` is code") == "`[04_memory/recall.py]` is code"


def test_a_bare_source_n_in_prose_is_a_citation_too(app):
    """Live: "The retry implementation in `Act.execute` is shown in Source 1." """
    assert app.linked("It is shown in Source 1.", {1: "u1"}) == "It is shown in [[1]](u1)."
    assert app.cited("It is shown in Source 1.", {1: "a.py, lines 1-5"}) == "It is shown in [a.py, lines 1-5]."
    assert app.linked("**Sources**", {1: "u1"}) == "**Sources**"                     # a heading, no number


def test_the_models_own_sources_list_is_dropped(app):
    answer = ("No, the retries are internal.\n\n**Sources**  \n- The retry implementation is shown in Source 1.\n"
              "- The definition of a decision is in Source 2.")
    assert app.without_own_sources(answer) == "No, the retries are internal."


def test_a_sources_heading_followed_by_prose_is_kept(app):
    answer = "### Sources\nThe two notebooks disagree here, and the code settles it."
    assert app.without_own_sources(answer) == answer


def test_passages_from_one_section_are_one_source_and_renumbered(app, monkeypatch):
    """Live: sources 2 and 4 were the same notebook section, listed twice. Merged, the answer's [4] becomes [2]."""
    from types import SimpleNamespace
    from ask_the_cookbook import corpus
    notebook = "02_agent_runtime/notebooks/02_the_agent_loop.ipynb"
    prose = corpus._notebook_prose(corpus.REPO / notebook).splitlines()
    at = prose.index("## `act.py`: executing a decision already made")
    first, second = [line for line in prose[at + 1:at + 40] if len(line.strip()) >= 30][:2]
    module = "02_agent_runtime/runtime.py"
    code = (corpus.REPO / module).read_text(encoding="utf-8").splitlines()
    start = next(n for n, line in enumerate(code, 1) if line.startswith("def _similar_args"))
    nodes = [passage(module, "\n".join(code[start - 1:start + 2])), passage(notebook, first),
             passage(module, "\n".join(code[start + 4:start + 6])), passage(notebook, second)]
    response = SimpleNamespace(source_nodes=[SimpleNamespace(node=node) for node in nodes])
    monkeypatch.setattr(app, "chat_model", lambda key: None)
    monkeypatch.setattr(app.A, "engine", lambda index, llm: None)
    monkeypatch.setattr(app, "rag", lambda q, key, stage: {
        "refused": False, "answer": "Code [1], notebook [2], more code [3], notebook again [4].", "response": response})
    shown, kept = app.ask_engine("How?", "gsk_x", Session())
    listed = shown.split("**Sources**")[1].strip().splitlines()
    assert len(listed) == 3                                         # the notebook section once
    assert "notebook again [[2]](" in shown and "[[4]]" not in shown
    assert kept.count(notebook) == 2                                # the chat keeps each passage's own lines


@pytest.mark.parametrize("answer, shown", [
    ("Ranked [1].", "Ranked [[1]](u1)."),
    ("Ranked [Source 2].", "Ranked [[2]](u2)."),
    ("Ranked [1, 2].", "Ranked [[1]](u1)[[2]](u2)."),
    ("Exact facts【1†L9-L15】.", "Exact facts[[1]](u1)."),
    ("A list [3] stays.", "A list [3] stays."),
])
def test_every_citation_form_is_shown_as_a_link(app, answer, shown):
    assert app.linked(answer, {1: "u1", 2: "u2"}) == shown


def test_compact_now_on_a_visitors_key_uses_their_key(app, monkeypatch, no_shared_calls):
    visitor = Compactor()
    monkeypatch.setattr(app, "client", lambda key: visitor)
    session = chat_of(6)
    shown, session, status, _ = list(app.compact_now("gsk_visitor", session))[-1]
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
    shown, session, status, _ = app.new_chat("")
    assert shown == [] and session.history == [] and session.notes == []
    assert "Chat holds **0 tokens**. The agent can take up to **5,000** per model call." in status
