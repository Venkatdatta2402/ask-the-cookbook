"""The page's answers: what is checked before a model is called, the daily limit, the engine's sources, the agent,
and errors in words. A stand-in LLM answers the engine; a fake client answers the agent."""

import os
import re

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
    assert app.DAILY.left() == app.DAILY_LIMIT - 1


def test_with_the_floor_a_question_nothing_is_near_is_turned_away(app, ask, monkeypatch):
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    reply, _ = ask("What is the capital of Australia?")
    assert reply.startswith(app.A.NOT_COVERED)


def test_the_days_count_survives_a_restart(app, tmp_path):
    """Visitor A takes 10; the app restarts; visitor B, in a new session, sees 40 -- not 50."""
    path = tmp_path / "daily_count.json"
    before = app.Daily(50, path)
    for _ in range(10):
        before.take()
    after = app.Daily(50, path)                                     # the restarted app reads the file
    assert after.left() == 40


def test_a_new_day_starts_the_count_again_and_a_bad_file_is_not_fatal(app, tmp_path):
    path = tmp_path / "daily_count.json"
    path.write_text('{"day": "2020-01-01", "used": 49}', encoding="utf-8")
    assert app.Daily(50, path).left() == 50                         # yesterday's count is not today's
    path.write_text("not json", encoding="utf-8")
    assert app.Daily(50, path).left() == 50


def test_the_sidebar_says_when_a_visitors_own_key_is_in_use(app, monkeypatch):
    """With a key in the box nothing is counted, so a count that never moved looked broken: the line says why."""
    from llama_index.core.llms import MockLLM
    from ask_the_cookbook.session import Session
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM(max_tokens=20))
    assert "plenty left today" in app.allowance(app.own(""))
    *_, line, _, _ = list(app.respond("What is a doom loop?", "", "gsk_visitor", app.RAG, Session()))[-1]
    assert line.startswith("Your key is in use: your questions go on your own Groq account, under its limits")
    assert app.DAILY.left() == app.DAILY_LIMIT                           # and nothing was counted
    *_, line, _, _ = list(app.respond("What is a doom loop?", "", "", app.RAG, Session()))[-1]
    assert "plenty left today" in line and app.DAILY.left() == app.DAILY_LIMIT - 1   # the demo key's is counted
    assert app.new_chat("gsk_visitor")[-1].startswith("Your key is in use")


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


@pytest.mark.parametrize("cited, link", [
    ("[02_agent_runtime/runtime.py]", "02_agent_runtime/runtime.py"),
    ("[ap_checks.py, line 4‑5]", "08_frameworks/02_langgraph/ap_checks.py#L4-L5"),
    ("[ap_checks.py, lines 4-8]", "08_frameworks/02_langgraph/ap_checks.py#L4-L8"),
    ("[04_memory/README.md § Memory, lines 3–5]", "04_memory/README.md?plain=1#L3-L5"),
    ("[ap_desk.ipynb, lines 348‑352]", "08_frameworks/02_langgraph/ap_desk.ipynb"),     # a notebook: from the top
])
def test_the_agents_file_citations_become_links_to_their_lines(app, cited, link):
    assert app.link_files(f"Shown {cited}.") == f"Shown [{cited}]({app.blob('')}{link})."


def test_the_agents_citations_in_gpt_oss_brackets_become_links_too(app):
    cited = "【08_frameworks/02_langgraph/ap_checks.py, lines 5‑6】"
    assert app.link_files(f"Twice {cited}.") == (
        f"Twice [[08_frameworks/02_langgraph/ap_checks.py, lines 5‑6]]"
        f"({app.blob('08_frameworks/02_langgraph/ap_checks.py')}#L5-L6).")


def test_gpt_oss_line_notation_reaches_the_link(app):
    """Live: 【08_frameworks/02_langgraph/case.py†L19-L24】."""
    linked = app.link_files("Ends the step 【08_frameworks/02_langgraph/case.py†L19-L24】.")
    assert linked.endswith(f"({app.blob('08_frameworks/02_langgraph/case.py')}#L19-L24).")


@pytest.mark.parametrize("text", [
    "[no_such_file.py, line 3]",            # not a cookbook file
    "[README](04_memory/README.md)",        # already a markdown link
    "values[1] and [the plan]",             # not citations at all
])
def test_what_is_not_a_cookbook_file_citation_is_left_alone(app, text):
    assert app.link_files(text) == text


def test_while_waiting_the_message_says_what_is_happening_with_turning_dots(app, monkeypatch):
    """RAG's two stages, as rag() reports them, reach the page while the answer is worked out."""
    import time

    def slow_rag(question, key, stage):
        stage["now"] = "Searching the cookbook"
        time.sleep(1.5)
        stage["now"] = "Reading 5 passages and writing the answer"
        time.sleep(1.5)
        return {"refused": True, "answer": app.A.NOT_COVERED, "response": None}

    monkeypatch.setattr(app, "rag", slow_rag)
    from ask_the_cookbook.session import Session
    waiting = [shown[-1]["content"] for shown, *_ in app.respond("q", "", "gsk_x", app.RAG, Session())][:-1]
    assert all(m.startswith('<span class="cooking" style="--phase: -') for m in waiting)   # the words, every time
    assert any('<span class="doing">Searching the cookbook</span>' in m for m in waiting)
    assert any('<span class="doing">Reading 5 passages and writing the answer</span>' in m for m in waiting)
    # re-sent only when what is happening changes: re-sending restarts the words' animation
    assert all(a != b for a, b in zip(waiting, waiting[1:])) and len(waiting) <= 3
    # one random order per question, the same in every re-drawn message
    orders = {tuple(re.findall(r"<span>([^<]*)…</span>", m)) for m in waiting}
    assert len(orders) == 1


def test_the_agents_step_is_said_as_a_visitor_would_say_it(app):
    from ask_the_cookbook.runtime.chat import Conversation
    doing = lambda stage: app.working(app.AGENTIC, stage).split('<span class="doing">')[1]
    conversation = Conversation(system="s")
    conversation.messages.append({"role": "decision", "type": "tool_call", "tool": "search_cookbook",
                                  "content": '{"query": "LangGraph <hazards>"}'})
    stage = {"conversation": conversation}
    assert doing(stage).startswith("Searching for “LangGraph &lt;hazards&gt;” · call 1 of up to 8")
    conversation.messages[-1]["result"] = "[...]"
    assert doing(stage).startswith("Read what call 1 found; choosing the next step")
    conversation.messages.append({"role": "decision", "type": "tool_call", "tool": "open_file",
                                  "content": '{"path": "08_frameworks/02_langgraph/ap_checks.py", "line_start": 1, '
                                             '"line_end": 120}'})
    assert doing(stage).startswith("Reading 08_frameworks/02_langgraph/ap_checks.py, lines 1–120 · call 2 of up")
    assert doing({}).startswith("Planning its first search")


def test_every_word_is_in_the_message_and_gets_its_own_slot(app):
    """The words are in the message itself, and only opacity and position are animated -- a first version swapped
    the text in CSS (`content`), and the browser showed nothing."""
    shown = re.findall(r"<span>([^<]*)…</span>", app.cooking())
    assert len(shown) == len(app.COOKING) * app.ROUNDS
    for n in range(len(shown)):
        assert (f".cooking > span:nth-child({n + 1}) {{ animation-delay: calc(var(--phase, 0s) + "
                f"{n * app.WORD_SECONDS}s)") in app.CSS
    assert "content:" not in app.COOKING_CSS


def test_the_words_come_in_a_random_order_every_word_each_round_never_twice_running(app):
    import random
    orders = [app.word_order(random.Random(seed)) for seed in range(20)]
    assert len({tuple(o) for o in orders}) == 20                                    # a different order each time
    for order in orders:
        size = len(app.COOKING)
        assert all(sorted(order[r * size:(r + 1) * size]) == sorted(app.COOKING) for r in range(app.ROUNDS))
        assert all(a != b for a, b in zip(order, order[1:]))                        # where rounds meet, too


def test_a_resent_message_carries_on_where_the_words_were(app):
    """Re-sent when the step changes, the message is drawn afresh: it carries how far into the cycle it is, so the
    word showing goes on showing instead of fading in again from the first."""
    assert 'style="--phase: -0.00s"' in app.working(app.RAG, {})
    assert 'style="--phase: -4.50s"' in app.working(app.RAG, {}, elapsed=4.5)
    assert 'style="--phase: -1.00s"' in app.working(app.RAG, {}, elapsed=app._CYCLE + 1)      # round the cycle


@pytest.mark.parametrize("taken, says", [(0, "plenty left today"), (39, "plenty left today"),
                                         (40, "running low"), (49, "running low"), (50, "used up for today")])
def test_the_allowance_says_how_much_is_left_not_a_number(app, taken, says):
    for _ in range(taken):
        app.DAILY.take()
    line = app.allowance()
    assert says in line and str(app.DAILY_LIMIT) not in line


def test_the_words_gradient_survives_gradios_rewrite_of_the_css(app):
    """Gradio rebuilds custom CSS from the browser's reading of it, and a `background` shorthand holding var() with
    longhands after it is dropped in that reading -- found rendering the page headless: the words were transparent
    ink on no background. The gradient must be a background-image longhand."""
    assert "background-image: linear-gradient(" in app.COOKING_CSS
    assert " background: " not in app.COOKING_CSS


def test_a_refused_rag_question_is_never_sent_to_the_model(app, monkeypatch):
    from llama_index.core.llms import MockLLM
    from llama_index.core.query_engine import CitationQueryEngine

    def never(*args, **kwargs):
        raise AssertionError("the answer was written for a question the floor turned away")
    monkeypatch.setattr(CitationQueryEngine, "synthesize", never)
    monkeypatch.setattr(app, "chat_model", lambda key: MockLLM())
    stage = {}
    result = app.rag("What is the capital of Australia?", "gsk_x", stage)        # the floor turns it away
    assert result["refused"] and result["answer"] == app.A.NOT_COVERED
    assert stage["now"] == "Searching the cookbook"                               # it never got to writing


def test_an_example_clicked_in_the_empty_chat_fills_the_question_box(app):
    from types import SimpleNamespace
    assert app.example(SimpleNamespace(value={"text": app.EXAMPLES[0]})) == app.EXAMPLES[0]


def test_the_page_loads(app):
    app.demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        assert httpx.get(app.demo.local_url).status_code == 200
    finally:
        app.demo.close()
