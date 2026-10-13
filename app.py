"""Ask the Cookbook, as a Gradio app: the Hugging Face Space.

    RAG           one retrieval, then one answer with citations (search.engine), on the owner's Groq key:
                  DAILY_LIMIT questions a day, all visitors together, counted here -- the count starts again if the
                  Space restarts, and Groq's own daily limit still stands behind it
    your own key  a Groq key (gsk_...): no daily limit, and agentic RAG -- retrieval as a tool the model calls as
                  often as it decides, in the agent loop (agent: chapter 2's loop, chapter 5's context). The same
                  model either way, gpt-oss-120b, the one the agent was tuned on. Groq only: Hugging Face's router
                  serves the model too, but bills from credit a free account no longer has (402 "no remaining
                  credits", measured), and a Groq key is free

It is a chat (session.py), kept for the browser session only. Select text in an answer and an "Ask about" button
appears: the passage goes into the question, which RAG retrieves with and agentic RAG has pinned in its context.
Without a selection, RAG answers the question as asked; agentic RAG has the earlier turns as candidates for each
call's context, and chapter 5's selection decides what fits. A meter shows how much the chat holds, and Compact
now replaces older turns with a summary (chapter 5's compaction). Every message, and every compaction, on the
owner's key counts against DAILY_LIMIT.

Every question also costs one Gemini embedding (an agent question one per search) on the owner's key, whichever
key answers it: the index was embedded with Gemini, and a question has to be too.

The cookbook's text is read from data/corpus (ask_the_cookbook.refresh writes it).
"""

import json
import os
import re
import threading
from datetime import datetime, timezone

import gradio as gr
import openai

from ask_the_cookbook import agent as S
from ask_the_cookbook import search as A
from ask_the_cookbook import session as chat
from ask_the_cookbook.session import Session

COOKBOOK = "https://github.com/Venkatdatta2402/agentic-forge-cookbook"
BUILT = json.loads((A.INDEX_DIR / "cookbook.json").read_text(encoding="utf-8"))

DAILY_LIMIT = 50            # questions a day on the owner's key: ~3,200 Groq tokens each, of 200k a day
MAX_QUESTION = 500          # characters: a question, not a document to summarise on someone else's key

RAG, AGENTIC = "RAG", "Agentic RAG"

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
    """A client for one agent run, or one compaction, alone: so calls at once on different keys stay apart."""
    return openai.OpenAI(api_key=key, base_url=A.GROQ, max_retries=8)


# --------------------------------------------------------------------------- answering

def blob(file):
    return f"{COOKBOOK}/blob/{BUILT['commit']}/{file}"


# A citation as gpt-oss writes it: [1], [1, 2], [Source 1], once live [Source 1, commentary in the same file] -- and
# its own trained form, 【1†L9-L15】 (source 1, lines 9-15 of the passage as it was shown). Each is placed by our own
# count of the passage's lines in its file, not by the model's.
_CITATION = re.compile(r"\[(?:Source\s*)?(\d+(?:\s*,\s*(?:Source\s*)?\d+)*)([^\[\]]*)\]|【(\d+)(?:†[^】]*)?】")


def citations(text, render):
    """Every numbered citation in `text` replaced by render(numbers, rest), where it returns something."""
    def one(match):
        numbers = [int(n) for n in re.findall(r"\d+", match[1] or match[3])]
        return render(numbers, match[2] or "") or match[0]
    return _CITATION.sub(one, text)


def cited(text, places):
    """The answer the chat keeps: each citation as "file, lines a-b", which means something later."""
    return citations(text, lambda numbers, rest: all(n in places for n in numbers)
                     and "[" + "; ".join(places[n] for n in numbers) + rest + "]")


def linked(text, links):
    """The answer the page shows: each citation as a small [n] that opens the cited lines on GitHub."""
    return citations(text, lambda numbers, rest: all(n in links for n in numbers)
                     and "".join(f"[[{n}]]({links[n]})" for n in numbers))


def span(node):
    """The lines of its file a passage covers, counted as the agent's open_file and search results count them (a
    notebook by its prose), or None if its text is no longer found there."""
    start = A.first_line(node)
    if start is None:
        return None
    lines = node.get_content().splitlines()
    # first_line() places the passage by its first line of 12 or more characters -- past the "Source 1:" header
    # CitationQueryEngine puts on each passage -- so the count starts there too
    body = lines[next(n for n, line in enumerate(lines) if len(line.strip()) >= 12):]
    while body and not body[-1].strip():
        body.pop()
    return start, start + len(body) - 1


def where(node):
    """A source passage as "file, lines a-b", for the chat to keep. Just the file if its lines cannot be placed."""
    file, lines = node.metadata["file"], span(node)
    return f"{file}, lines {lines[0]}-{lines[1]}" if lines else file


def heading_above(file, line):
    """The heading a passage sits under: the nearest markdown heading at or above its first line, in the file's text
    as indexed (a notebook's prose). Its own `header_path` will not do -- it lists the headings ABOVE the passage's
    section, and a passage that continues a section has no heading of its own."""
    heading, fenced = None, False
    for text in A.file_text(file).splitlines()[:line]:
        if text.lstrip().startswith("```"):
            fenced = not fenced                     # a "# comment" in a code example is not a heading
        elif not fenced and re.match(r"#{1,6} ", text):
            heading = text.lstrip("#").strip()
    return heading


def source(node):
    """(how the page names a source passage, the link that opens it). A .py or .md file opens on GitHub at the
    passage's lines -- a README in its raw form, the one GitHub can point at a line in. A notebook opens on GitHub
    from the top: GitHub shows notebooks in a frame no link can scroll, and nbviewer, which could, was refused by
    GitHub's rate limit when tried live. It is named by the heading the passage sits under, to find it by, and
    carries no line numbers: ours count its prose as indexed, which GitHub does not show as such."""
    file, lines = node.metadata["file"], span(node)
    heading = heading_above(file, lines[0]) if lines and not file.endswith(".py") else None
    section = f" § {heading}" if heading else ""
    if file.endswith(".ipynb"):
        return file + section, blob(file)
    if not lines:
        return file, blob(file)
    raw = "?plain=1" if file.endswith(".md") else ""
    return f"{file}, lines {lines[0]}-{lines[1]}{section}", f"{blob(file)}{raw}#L{lines[0]}-L{lines[1]}"


FOLLOW_UP = ("To follow up on an earlier answer, select the part you mean and press **Ask about**, or ask a question "
             "that names the chapter or idea.")


def ask_engine(question, key, session):
    """(what the page shows, what the chat keeps). Retrieval runs on the question as asked -- with the passage, when
    the visitor asked about one. The chat keeps the answer with its numbered citations turned into "file, lines a-b",
    which mean something later, and are lines a later agentic turn can open."""
    result = A.ask(A.engine(INDEX, chat_model(key)), question)
    if result["refused"]:
        text = f"{A.NOT_COVERED} Nothing in it is close enough to the question to answer from."
        # a follow-up turned away is most often one that leaned on the chat: retrieval saw only its own words
        return (f"{text}\n\n{FOLLOW_UP}" if session.history else text), text
    nodes = dict(enumerate((s.node for s in result["response"].source_nodes), 1))
    named = {n: source(node) for n, node in nodes.items()}
    sources = "\n".join(f"{n}. [{name}]({url})" for n, (name, url) in named.items())
    shown = linked(result["answer"], {n: url for n, (_, url) in named.items()})
    kept = cited(result["answer"], {n: where(node) for n, node in nodes.items()})
    return f"{shown}\n\n**Sources**\n\n{sources}", kept


_CITED = re.compile(r"\[(\d\d_[^\[\]\s]+?\.(?:py|md|ipynb)|README\.md)\]")


def ask_agent(question, key, session):
    """(what the page shows, what the chat keeps). The run starts from the chat so far."""
    own = client(key)
    send, refused = own.chat.completions.create, []

    # agent.run reports a refused request as why the run stopped, in words; the error itself is kept here, so a bad
    # key or a spent limit can be told apart. The client is this run's alone, so this touches no other run
    def create(*args, **kwargs):
        try:
            return send(*args, **kwargs)
        except openai.APIStatusError as error:
            refused.append(error)
            raise

    own.chat.completions.create = create
    result = S.run(INDEX, question, A.Budget(label="agent"), client=own, history=session.history, notes=session.notes)
    answer = result["answer"]
    if not answer:
        if (result["stopped"] or "").startswith("BudgetExceeded"):
            text = "The agent used its token budget before it had an answer. Try a narrower question."
            return text, text
        if refused:
            raise refused[-1]
        raise RuntimeError(f"the agent stopped without an answer ({result['stopped']})")
    linked = _CITED.sub(lambda m: f"[[{m[1]}]]({blob(m[1])})", answer)
    steps = "\n".join(f"- `{c['tool']}` {str(c['args'])[:150]}" for c in result["calls"])
    usage = result["usage"]
    return (f"{linked}\n\n<details><summary>How it got there: {len(result['calls'])} tool calls, "
            f"{usage['total']:,} tokens on your key</summary>\n\n{steps}\n\n</details>"), answer


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
        return "The demo's Groq key has reached its daily limit. Paste your own key in the sidebar, or come back tomorrow."
    if isinstance(error, openai.APIStatusError) and status == 413:
        return "That request was too large for the model's limit. Try a shorter or narrower question."
    print(f"error: {type(error).__name__}: {str(error)[:300]}", flush=True)
    return "Something went wrong answering that. Please try again."


def status(session, note=""):
    """The line under the chat: a note when there is one, then how much the chat holds."""
    return ((f"{note}\n\n" if note else "")
            + f"Chat holds **{session.held():,} tokens**. The agent can take up to **{chat.AGENT_CALL:,}** per model "
              f"call.")


def allowance():
    """The sidebar's line on the demo key, beside the key box it is the alternative to."""
    return f"Without a key: **{DAILY.left()} of {DAILY_LIMIT}** free RAG questions left today, shared by all visitors."


def whose_key(key, mode=None):
    """(the key to use, own key?, a problem to show instead), taking a daily question when it is the demo key's."""
    key = (key or "").strip()
    if key:
        if not key.startswith("gsk_"):
            return None, True, "That isn't a Groq API key: those start `gsk_`."
        return key, True, None
    if mode == AGENTIC:
        return None, False, "Agentic RAG runs on your own key: paste a Groq API key in the sidebar."
    if not DAILY.take():
        return None, False, ("Today's free questions are used up. Paste your own Groq API key in the sidebar to keep "
                             "asking, or come back tomorrow.")
    return os.environ["GROQ_API_KEY"], False, None


def respond(question, quote, key, mode, session):
    """One message, from the page: a question, and the passage it is about if the visitor pressed Ask about.
    Yields (the chat, the session, the status line, the demo key's allowance, the question box, the quote box)."""
    question, quote = (question or "").strip(), (quote or "").strip()

    def turned_away(note, keep_question=True):
        return session.shown, session, status(session, note), allowance(), question if keep_question else "", quote

    if not question:
        yield turned_away("Type a question first.", keep_question=False)
        return
    if len(question) > MAX_QUESTION:
        yield turned_away(f"Please keep the question under {MAX_QUESTION} characters.")
        return
    if len(quote) > chat.QUOTE_LIMIT:
        yield turned_away(f"Please select at most {chat.QUOTE_LIMIT:,} characters to ask about "
                          f"(this is {len(quote):,}).")
        return
    key, own_key, problem = whose_key(key, mode)
    if problem:
        yield turned_away(problem)
        return

    working = ("The agent is searching the cookbook. This takes a few minutes: on Groq's free tier its calls wait "
               "for the per-minute token limit." if mode == AGENTIC else "Searching the cookbook...")
    session.shown += [{"role": "user", "content": chat.showing(question, quote)},
                      {"role": "assistant", "content": f"*{working}*"}]
    yield session.shown, session, status(session), allowance(), "", ""
    message = chat.asking(question, quote)
    try:
        shown, kept = (ask_agent if mode == AGENTIC else ask_engine)(message, key, session)
        session.add(message, kept)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        shown = trouble(error, own_key)
    session.shown[-1] = {"role": "assistant", "content": shown}
    yield session.shown, session, status(session), allowance(), "", ""


def compact_now(key, session):
    """Compact now: older turns replaced by a summary and facts, by chapter 5's compaction. Yields (the chat, the
    session, the status line, the demo key's allowance). On the demo key it counts as a question; on a visitor's key
    the calls go on theirs."""
    if not session.can_compact():
        note = (f"Not worth compacting yet: {session.compactable():,} tokens are older than the last "
                f"{chat.KEEP_RECENT // 2} exchanges (which are always kept as they are), and a summary is only "
                f"smaller once at least {chat.COMPACT_WORTH:,} are.")
        yield session.shown, session, status(session, note), allowance()
        return
    key, own_key, problem = whose_key(key)
    if problem:
        yield session.shown, session, status(session, problem), allowance()
        return
    yield session.shown, session, status(session, "*Compacting older turns...*"), allowance()
    try:
        before, after = session.compact(client(key) if own_key else None)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        yield session.shown, session, status(session, trouble(error, own_key)), allowance()
        return
    if after >= before:
        note = "Compacting would not have made the chat any smaller, so it was left as it was."
        yield session.shown, session, status(session, note), allowance()
        return
    note = (f"*Older turns compacted: the chat now holds {after:,} tokens, down from {before:,}. The transcript above "
            "stays; the model gets a summary of the older part.*")
    session.shown.append({"role": "assistant", "content": note})
    yield session.shown, session, status(session), allowance()


def new_chat():
    fresh = Session()
    return fresh.shown, fresh, status(fresh), allowance()


def example(choice: gr.SelectData):
    """An example question, clicked in the empty chat, into the question box -- not sent, so the mode can be chosen."""
    return choice.value["text"]


# --------------------------------------------------------------------------- the page

EXAMPLES = [
    "What counts as a doom loop, and what does the runtime do when it detects one?",
    "Why shouldn't agent memory start with a vector database?",
    "What is the difference between handing off a conversation and delegating to another agent?",
    "Why is CrewAI pinned to an old version in this repo?",
]

# "Ask about": select text in the chat and a small button appears beside it; pressing it puts the selection in the
# Asking about box. The page script keeps the selection and presses a hidden button whose event copies it into the box
# (an event's `js` with no Python function fills its outputs straight from the browser).
ASK_ABOUT_JS = """
() => {
  let chosen = "";
  const chip = document.createElement("button");
  chip.id = "ask-about-chip";
  chip.type = "button";
  chip.textContent = "Ask about \\u2197";
  chip.style.display = "none";
  document.body.appendChild(chip);

  const inChat = (node) => {
    const chat = document.getElementById("chat");
    return Boolean(chat && node && chat.contains(node.nodeType === 1 ? node : node.parentNode));
  };
  const hide = () => { chip.style.display = "none"; };

  document.addEventListener("mouseup", (event) => {
    if (event.target === chip) return;
    setTimeout(() => {                                   // after the selection has settled
      const selection = window.getSelection();
      const text = selection ? selection.toString().trim() : "";
      if (!text || !inChat(selection.anchorNode) || !inChat(selection.focusNode)) { hide(); return; }
      chosen = text;
      const box = selection.getRangeAt(0).getBoundingClientRect();
      chip.style.top = `${box.bottom + 6}px`;
      chip.style.left = `${Math.max(8, Math.min(box.right - 40, window.innerWidth - 130))}px`;
      chip.style.display = "block";
    }, 0);
  });
  document.addEventListener("scroll", hide, true);
  chip.addEventListener("mousedown", (event) => event.preventDefault());   // keep the selection while clicking
  chip.addEventListener("click", () => {
    window.askAboutText = chosen;
    hide();
    const trigger = document.getElementById("ask-about-trigger");
    (trigger.tagName === "BUTTON" ? trigger : trigger.querySelector("button")).click();
    window.getSelection().removeAllRanges();
    setTimeout(() => document.querySelector("#question textarea")?.focus(), 100);
  });
}
"""

CSS = """
#ask-about-chip {
  position: fixed; z-index: 1000; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
  border: 1px solid var(--button-primary-border-color); background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
#ask-about-trigger { display: none !important; }
/* the passage being asked about shows only once there is one */
#quote-box:has(textarea:placeholder-shown) { display: none !important; }
#quote-box { border-left: 3px solid var(--color-accent); }
/* the chat's state, quiet and on one line with the two things that act on it */
#chat-state { align-items: center; }
#meter, #meter p { font-size: var(--text-sm); color: var(--body-text-color-subdued); margin: 0; }
#allowance, #index { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="Ask the Cookbook") as demo:
    session = gr.State(Session())

    with gr.Sidebar(width=320):
        gr.Markdown(f"## Ask the Cookbook\nQuestions about [agentic-forge-cookbook]({COOKBOOK}), answered from its "
                    "READMEs, notebooks and code, with the files cited.")
        mode = gr.Radio([RAG, AGENTIC], value=RAG, label="Mode",
                        info="RAG: one search, an answer in seconds. Agentic RAG: searches as many times as the "
                             "question needs and reads files, for questions that span chapters; a few minutes.")
        key = gr.Textbox(label="Your Groq API key", type="password", placeholder="gsk_...",
                         info="Needed for Agentic RAG, and with it no daily limit. Used for your questions only, "
                              "never stored.")
        gr.Markdown("Get one free at [console.groq.com/keys](https://console.groq.com/keys).")
        allowance_line = gr.Markdown(elem_id="allowance")
        gr.Markdown(f"Index: the cookbook at [`{BUILT['commit'][:7]}`]({COOKBOOK}/tree/{BUILT['commit']}), "
                    f"{BUILT['files']} files. The chat lasts for this browser session; nothing is stored.",
                    elem_id="index")

    conversation = gr.Chatbot(show_label=False, height="62vh", elem_id="chat",
                              examples=[{"text": q} for q in EXAMPLES],
                              placeholder="**Ask anything about the cookbook.** To follow up on part of an answer, "
                                          "select it and press **Ask about**.")
    with gr.Row(elem_id="chat-state"):
        meter = gr.Markdown(elem_id="meter")
        compact = gr.Button("Compact now", size="sm", scale=0, min_width=120)
        clear = gr.Button("New chat", size="sm", scale=0, min_width=100)
    with gr.Group(elem_id="quote-box"):
        with gr.Row():
            quote = gr.Textbox(label="Asking about", placeholder=" ", lines=1, max_lines=6, scale=8,
                               elem_id="quote")
            gr.ClearButton(quote, value="✕", size="sm", scale=0, min_width=40)
    question = gr.Textbox(show_label=False, placeholder="Ask about the cookbook...", lines=1, max_lines=6,
                          submit_btn=True, elem_id="question")
    ask_about = gr.Button("Ask about", elem_id="ask-about-trigger")

    # private: not offered to API clients, so the demo key's daily questions are asked through this page
    question.submit(respond, [question, quote, key, mode, session],
                    [conversation, session, meter, allowance_line, question, quote],
                    concurrency_limit=8, api_visibility="private")
    ask_about.click(None, None, quote, js="() => window.askAboutText || ''", api_visibility="private")
    conversation.example_select(example, None, question, api_visibility="private")
    compact.click(compact_now, [key, session], [conversation, session, meter, allowance_line], concurrency_limit=8,
                  api_visibility="private")
    clear.click(new_chat, None, [conversation, session, meter, allowance_line], api_visibility="private")
    demo.load(lambda s: (status(s), allowance()), session, [meter, allowance_line], api_visibility="private")

demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch(js=ASK_ABOUT_JS, css=CSS, theme=gr.themes.Soft())
