"""The agent's calls note: every tool call made so far, with the labels of what it returned, in every model call."""

import json

import pytest
from openai.types.chat import ChatCompletion

from ask_the_cookbook import agent, search
from ask_the_cookbook.runtime.chat import Conversation

# different enough that chapter 2's doom-loop check (same tool, similar arguments, same result) does not stop a run
QUERIES = ["memory stores and recall ranking", "doom loop detection in the runtime",
           "context selection within a token budget", "LangGraph interrupt and resume", "tool registry schemas"]


@pytest.mark.parametrize("earlier, query", [
    # the three live runs' second searches, and the other near-copies they made
    ("LangGraph desk wait for a person case file wait for a person", "desk wait for a person LangGraph"),
    ("LangGraph desk wait for a person", "waiting for a person LangGraph desk"),
    ("hazard LangGraph desk", "LangGraph hazards"),
    ("double_send LangGraph", "double_send"),
])
def test_a_search_in_other_words_is_a_near_duplicate(earlier, query):
    assert search.near_duplicate(query, ["something else entirely", earlier]) == earlier


@pytest.mark.parametrize("earlier, query", [
    # searches the live runs made that DID look for something new
    ("LangGraph desk wait for a person case file wait for a person", "The same desk, from scratch case file wait"),
    ("LangGraph hazards", "CaseFile wait person Wait class case file step returns Wait"),
    ("chapter 5 context selection", "doom loop context selection"),
])
def test_a_search_for_something_new_is_not(earlier, query):
    assert search.near_duplicate(query, [earlier]) is None


def test_the_search_tool_refuses_a_near_duplicate_and_says_why(app):
    from llama_index.core.llms import MockLLM
    search_cookbook, _ = search.tool_functions(app.INDEX, MockLLM())
    first = search_cookbook("LangGraph desk wait for a person")
    again = search_cookbook("waiting for a person LangGraph desk")
    assert first.startswith("[")                                        # passages, labelled
    assert again.startswith('You already searched for nearly this: "LangGraph desk wait for a person"')
    assert search_cookbook("doom loop detection in the runtime").startswith("[")


def test_no_calls_no_note():
    assert agent.calls_note(Conversation(system="s")) is None


def test_the_note_lists_each_call_with_the_labels_of_what_it_returned():
    conversation = Conversation(system="s")
    conversation.messages += [
        {"role": "user", "content": "How does a LangGraph node wait?"},
        {"role": "decision", "type": "tool_call", "tool": "search_cookbook",
         "content": '{"query": "double_send LangGraph"}', "applied": True,
         "result": "[08_frameworks/02_langgraph/ap_checks.py, lines 4-8]\n    double_send ...\n\n"
                   "[08_frameworks/02_langgraph/ap_desk.ipynb § Hazards, lines 680-690]\n**A resumed node...**"},
        {"role": "tool", "content": "(the same, verbatim)"},
        {"role": "decision", "type": "tool_call", "tool": "search_cookbook",
         "content": '{"query": "double_send LangGraph"}', "applied": True,
         "result": "You already made exactly this call; what it returned is above, or was cut to its labels to keep "
                   "requests small. Answer from what you have found, or try something different."},
        {"role": "tool", "content": "(the same, verbatim)"},
        {"role": "decision", "type": "tool_call", "tool": "search_cookbook",
         "content": '{"query": "weather in Paris"}', "applied": True, "result": "Nothing in the cookbook is close to that."},
    ]
    note = agent.calls_note(conversation)
    assert note.pinned and note.kind == "state"
    lines = note.content.splitlines()
    assert lines[1] == ('1. search_cookbook {"query": "double_send LangGraph"} -> '
                        "[08_frameworks/02_langgraph/ap_checks.py, lines 4-8]; "
                        "[08_frameworks/02_langgraph/ap_desk.ipynb § Hazards, lines 680-690]")
    assert lines[2].startswith('2. search_cookbook {"query": "double_send LangGraph"} -> You already made exactly')
    assert lines[3] == '3. search_cookbook {"query": "weather in Paris"} -> Nothing in the cookbook is close to that.'
    assert "Do not repeat a call, and do not search again with mostly the same words" in lines[0]


class Searcher:
    """A client that searches with a new query every time tools are offered, and answers when they are not -- the
    step-limit answer's call. Records every request."""

    def __init__(self):
        self.sent, self.chat = [], self
        self.completions = self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        if "tools" in kwargs:
            n = len(self.sent)
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"c{n}", "type": "function",
                "function": {"name": "search_cookbook", "arguments": json.dumps({"query": QUERIES[n - 1]})}}]}
        else:
            message = {"role": "assistant", "content": "What the searches found."}
        return ChatCompletion.model_validate({
            "id": "f", "object": "chat.completion", "created": 0, "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})


def tokens_of(seen):
    return sum(int(line.split()[1].rstrip("t")) for line in seen)


def test_the_step_limit_answer_gets_more_room_and_no_calls_note(app):
    """Ordinary calls cut to 1,800 tokens here: room for the question and the last two results (~1,750), not three.
    Three searches run (the limit's fourth call is decided, never run); the answer the step limit forces gets
    FINAL_TOKENS -- all three -- and only what it can actually read, no labels of what it cannot."""
    client = Searcher()
    result = agent.run(app.INDEX, "How do memory stores work?", search.Budget(label="test"), client=client,
                       context_tokens=1800, max_tool_calls=4)
    *calls, final = client.sent
    assert "tools" not in final and result["answer"] == "What the searches found."
    assert result["stopped"] == "reached the maximum of 4 tool calls"
    text = lambda request: " ".join(str(m.get("content")) for m in request["messages"])
    assert "Tool calls already made" in text(calls[1])
    assert "Tool calls already made" not in text(final)
    observations = [line for line in result["last call"] if line.startswith("observation")]
    assert len(observations) == 3 and tokens_of(result["last call"]) > 1800


def test_when_the_must_keeps_leave_no_room_for_the_note_the_call_goes_without_it(app):
    """The last two results fit the budget, but not beside the note: the run goes on, that call without the note --
    it stopped with "selection refused" before."""
    client = Searcher()
    result = agent.run(app.INDEX, "How do memory stores work?", search.Budget(label="test"), client=client,
                       context_tokens=1800, max_tool_calls=4)
    assert not (result["stopped"] or "").startswith("selection refused")
    text = " ".join(str(m.get("content")) for m in client.sent[3]["messages"])     # the 4th call: 3 results so far
    assert "Tool calls already made" not in text


def test_the_run_is_logged_without_any_text(app, capsys):
    client = Searcher()
    result = agent.run(app.INDEX, "How do memory stores work?", search.Budget(label="test"), client=client,
                       max_tool_calls=1)
    app.log_run(result)
    log = capsys.readouterr().out
    assert "agentic run: 1 tool calls" in log and "last call was given:" in log
    assert "tokens in / out" in log and "/1" in log                    # each call's input, and its output (1)
    assert "call 1:" in log and "tool: search_cookbook" in log
    assert "What the searches found" not in log                      # the answer is not logged


def test_every_model_call_after_the_first_carries_the_calls_so_far(app):
    """A whole run on a fake client: three searches, then an answer. The third request must list the first two."""
    sent = []

    class Client:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **kwargs):
            sent.append(kwargs["messages"])
            n = len(sent)
            if n <= 3:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"c{n}", "type": "function",
                    "function": {"name": "search_cookbook", "arguments": json.dumps({"query": QUERIES[n - 1]})}}]}
            else:
                message = {"role": "assistant", "content": "Done."}
            return ChatCompletion.model_validate({
                "id": "f", "object": "chat.completion", "created": 0, "model": kwargs["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})

    result = agent.run(app.INDEX, "How do memory stores work?", search.Budget(label="test"), client=Client())
    assert result["answer"] == "Done."
    first, third = (" ".join(str(m.get("content")) for m in sent[i]) for i in (0, 2))
    assert "Tool calls already made" not in first
    assert f'1. search_cookbook {{"query":"{QUERIES[0]}"}}' in third.replace(": ", ":")
    assert f'2. search_cookbook {{"query":"{QUERIES[1]}"}}' in third.replace(": ", ":")
