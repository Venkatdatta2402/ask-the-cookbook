"""Ask the Cookbook, as a Gradio app: the Hugging Face Space.

    default       the one-shot engine (ask_index.engine) on the owner's Groq key: DAILY_LIMIT questions a day, all
                  visitors together, counted here -- the count starts again if the Space restarts, and Groq's own
                  daily limit still stands behind it
    your own key  a Groq key (gsk_...): no daily limit, and the agent (agent_scratch: chapter 2's loop, chapter 5's
                  context). The same model either way, gpt-oss-120b, the one the agent was tuned on. Groq only:
                  Hugging Face's router serves the model too, but bills from credit a free account no longer has
                  (402 "no remaining credits", measured), and a Groq key is free

Every question also costs one Gemini embedding (an agent question one per search) on the owner's key, whichever
key answers it: the index was embedded with Gemini, and a question has to be too.

The cookbook's text is read from corpus/ (refresh_index.py writes it), not from this repo, which holds only the
modules the assistant runs on.
"""

import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "08_frameworks" / "05_llamaindex"))

import gradio as gr  # noqa: E402
import openai  # noqa: E402

import agent_scratch as S  # noqa: E402  -- before CB.REPO moves: it puts the chapters on sys.path from CB.REPO
import ask_index as A  # noqa: E402
import cookbook as CB  # noqa: E402

CORPUS = ROOT / "corpus"
CB.REPO, CB.SELF = CORPUS, CORPUS / "08_frameworks" / "05_llamaindex"

COOKBOOK = "https://github.com/Venkatdatta2402/agentic-forge-cookbook"
BUILT = json.loads((A.INDEX_DIR / "cookbook.json").read_text(encoding="utf-8"))

DAILY_LIMIT = 50            # questions a day on the owner's key: ~3,200 Groq tokens each, of 200k a day
MAX_QUESTION = 500          # characters: a question, not a document to summarise on someone else's key

QUICK, AGENT = "Quick answer", "Agent (your own key)"

INDEX = A.open_index()


# --------------------------------------------------------------------------- the daily limit

class Daily:
    """Questions answered on the owner's key today (UTC), and none past the limit. Taken before the question is
    answered, so visitors asking at once cannot go past it together."""

    def __init__(self, limit):
        self.limit, self.day, self.used = limit, None, 0
        self.lock = threading.Lock()

    def _today(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.used = today, 0

    def take(self):
        with self.lock:
            self._today()
            if self.used >= self.limit:
                return False
            self.used += 1
            return True

    def left(self):
        with self.lock:
            self._today()
            return self.limit - self.used


DAILY = Daily(DAILY_LIMIT)


# --------------------------------------------------------------------------- the models

_SWAP = threading.Lock()


def chat_model(key):
    """ask_index.chat_model on any Groq key. It reads the key from GROQ_API_KEY when it is called and takes none as
    an argument -- the copied code stays as it is -- so the key is set for the call and put back, under a lock.
    The clients it makes keep the key they were given."""
    with _SWAP:
        saved = os.environ["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = key
        try:
            return A.chat_model(budget=A.Budget(label="engine"))
        finally:
            os.environ["GROQ_API_KEY"] = saved


def client(key):
    """A client for one agent run alone: agent_scratch.run takes it, so runs at once on different keys stay apart."""
    return openai.OpenAI(api_key=key, base_url=A.GROQ, max_retries=8)


# --------------------------------------------------------------------------- answering

def blob(file):
    return f"{COOKBOOK}/blob/{BUILT['commit']}/{file}"


def ask_engine(question, key):
    result = A.ask(A.engine(INDEX, chat_model(key)), question)
    if result["refused"]:
        return f"{A.NOT_COVERED} Nothing in it is close enough to the question to answer from."
    sources = "\n".join(f"{n}. [{A.label(s.node)}]({blob(s.node.metadata['file'])})"
                        for n, s in enumerate(result["response"].source_nodes, 1))
    return f"{result['answer']}\n\n**Sources**\n\n{sources}"


_CITED = re.compile(r"\[(\d\d_[^\[\]\s]+?\.(?:py|md|ipynb)|README\.md)\]")


def ask_agent(question, key):
    own = client(key)
    send, refused = own.chat.completions.create, []

    # agent_scratch.run reports a refused request as why the run stopped, in words; the error itself is kept here,
    # so a bad key or a spent limit can be told apart. The client is this run's alone, so this touches no other run
    def create(*args, **kwargs):
        try:
            return send(*args, **kwargs)
        except openai.APIStatusError as error:
            refused.append(error)
            raise

    own.chat.completions.create = create
    result = S.run(INDEX, question, A.Budget(label="agent"), client=own)
    answer = result["answer"]
    if not answer:
        if (result["stopped"] or "").startswith("BudgetExceeded"):
            return "The agent used its token budget before it had an answer. Try a narrower question."
        if refused:
            raise refused[-1]
        raise RuntimeError(f"the agent stopped without an answer ({result['stopped']})")
    answer = _CITED.sub(lambda m: f"[[{m[1]}]]({blob(m[1])})", answer)
    steps = "\n".join(f"- `{c['tool']}` {str(c['args'])[:150]}" for c in result["calls"])
    usage = result["usage"]
    return (f"{answer}\n\n<details><summary>How it got there: {len(result['calls'])} tool calls, "
            f"{usage['total']:,} tokens on your key</summary>\n\n{steps}\n\n</details>")


def trouble(error, own_key):
    """What went wrong, in words a visitor can act on. Never the key, and never a stack trace."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    gemini = type(error).__module__.startswith("google")
    if gemini and status == 429:
        return "The search has used up its quota for today (Gemini's free tier). Please come back tomorrow."
    if isinstance(error, openai.AuthenticationError) or status in (401, 403):
        return "That key was refused. Check that it is a current Groq API key."
    if isinstance(error, openai.RateLimitError):
        if own_key:
            return "Your key has hit its rate limit. Wait a little, or until its daily limit resets."
        return "The demo's Groq key has reached its daily limit. Paste your own key below, or come back tomorrow."
    if isinstance(error, openai.APIStatusError) and status == 413:
        return "That request was too large for the model's limit. Try a shorter or narrower question."
    print(f"error: {type(error).__name__}: {str(error)[:300]}", flush=True)
    return "Something went wrong answering that. Please try again."


def left_today():
    return (f"Free questions left today on the demo key: **{DAILY.left()} of {DAILY_LIMIT}**. "
            f"Index: the cookbook at [`{BUILT['commit'][:7]}`]({COOKBOOK}/tree/{BUILT['commit']}), "
            f"{BUILT['files']} files.")


def respond(question, key, mode):
    """One question, from the page: yields the answer box and the line under it, first a working note."""
    question, key = (question or "").strip(), (key or "").strip()
    own_key = bool(key)
    if not question:
        yield "Type a question first.", left_today()
        return
    if len(question) > MAX_QUESTION:
        yield f"Please keep the question under {MAX_QUESTION} characters.", left_today()
        return
    if key:
        if not key.startswith("gsk_"):
            yield "That isn't a Groq API key: those start `gsk_`.", left_today()
            return
    elif mode == AGENT:
        yield "The agent runs on your own key: paste a Groq API key below.", left_today()
        return
    elif not DAILY.take():
        yield ("Today's free questions are used up. Paste your own Groq API key below to keep asking, or come "
               "back tomorrow."), left_today()
        return
    else:
        key = os.environ["GROQ_API_KEY"]

    try:
        if mode == AGENT:
            yield ("The agent is searching the cookbook. This takes a few minutes: on Groq's free tier its calls "
                   "wait for the per-minute token limit."), left_today()
            answer = ask_agent(question, key)
        else:
            yield "Searching the cookbook...", left_today()
            answer = ask_engine(question, key)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        answer = trouble(error, own_key)
    yield answer, left_today()


# --------------------------------------------------------------------------- the page

EXAMPLES = [
    "What counts as a doom loop, and what does the runtime do when it detects one?",
    "Why shouldn't agent memory start with a vector database?",
    "What is the difference between handing off a conversation and delegating to another agent?",
    "Why is CrewAI pinned to an old version in this repo?",
]

with gr.Blocks(title="Ask the Cookbook") as demo:
    gr.Markdown(f"# Ask the Cookbook\nQuestions about [agentic-forge-cookbook]({COOKBOOK}), answered from its "
                "READMEs, notebooks and code, with the files cited.")
    with gr.Row():
        question = gr.Textbox(label="Your question", lines=2, max_lines=6, scale=5)
        ask = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=question)
    answer = gr.Markdown()
    with gr.Accordion("Use your own key: no daily limit, and the agent", open=False):
        gr.Markdown("A [Groq API key](https://console.groq.com/keys) (free). It is used for your questions only and is "
                    "not stored. The agent searches more than once and reads files, for questions that span chapters: "
                    "it uses 30-40k tokens a question.")
        key = gr.Textbox(label="Groq API key", type="password", placeholder="gsk_...")
        mode = gr.Radio([QUICK, AGENT], value=QUICK, label="Mode")
    status = gr.Markdown()

    # private: not offered to API clients, so the demo key's daily questions are asked through this page
    for event in (ask.click, question.submit):
        event(respond, [question, key, mode], [answer, status], concurrency_limit=8, api_visibility="private")
    demo.load(left_today, None, status, api_visibility="private")

demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch()
